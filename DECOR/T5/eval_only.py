import argparse
import os
from transformers import EarlyStoppingCallback

import torch
import transformers
from transformers import T5Config

from modeling_decor import DECOR
from test import run_generation_eval
from utils import *
from collator import Collator


def get_valid_metric_name(args):
    if args.valid_metric == "loss":
        return "loss"
    if args.valid_metric == "ndcg":
        return f"ndcg@{args.valid_metric_k}"
    if args.valid_metric == "hitrate":
        return f"hitrate@{args.valid_metric_k}"
    raise ValueError(f"Unsupported valid metric: {args.valid_metric}")


class DecorTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        optimizer_impl="adamw_torch",
        valid_metric_mode="loss",
        valid_metric_name="loss",
        valid_eval_args=None,
        valid_tokenizer=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_impl = optimizer_impl
        self.valid_metric_mode = valid_metric_mode
        self.valid_metric_name = valid_metric_name
        self.valid_eval_args = valid_eval_args
        self.valid_tokenizer = valid_tokenizer
        self._last_breakdown_log_step = -1

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.valid_metric_mode == "loss" and hasattr(eval_dataset, "refresh_eval_samples"):
            eval_dataset.refresh_eval_samples()

        if self.valid_metric_mode == "loss":
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        self._memory_tracker.start()
        if eval_dataset is None:
            raise ValueError("Evaluation requires a validation dataset.")

        results = run_generation_eval(
            self.valid_eval_args,
            self.model,
            self.valid_tokenizer,
            eval_dataset,
            self.args.device,
            results_file=None,
            use_trie_override=False,
            metrics_override=self.valid_metric_name,
            prompt_ids_override=[self.valid_eval_args.valid_prompt_id],
            verbose=False,
        )
        metric_key = f"{metric_key_prefix}_{self.valid_metric_name}"
        metrics = {metric_key: results["mean_results"][self.valid_metric_name]}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if model.training and self.args.logging_steps > 0:
            next_step = self.state.global_step + 1
            if (
                next_step % self.args.logging_steps == 0
                and next_step != self._last_breakdown_log_step
            ):
                unwrapped_model = model.module if hasattr(model, "module") else model
                breakdown = getattr(unwrapped_model, "latest_loss_breakdown", None)
                if breakdown:
                    self.log(
                        {
                            key: value.detach().float().mean().item()
                            if torch.is_tensor(value)
                            else float(value)
                            for key, value in breakdown.items()
                        }
                    )
                    self._last_breakdown_log_step = next_step

        return (loss, outputs) if return_outputs else loss


def evaluate_only(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    device = torch.device("cuda", local_rank)

    train_data, valid_data = load_datasets(args)
    test_data = load_test_dataset(args)
    config_source = args.ckpt_path if args.ckpt_path and os.path.isdir(args.ckpt_path) else args.base_model
    config = T5Config.from_pretrained(config_source, local_files_only=True)
    tokenizer, add_num = build_tokenizer(
        args,
        model_max_length=args.model_max_length,
        new_tokens=train_data.datasets[0].get_new_tokens(),
        tokenizer_source=config_source,
        local_files_only=True,
    )
    pretokenize_datasets(tokenizer, train_data, valid_data, test_data)
    config.mask_token_id = -1
    config.maskgit_target_length = 4
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        print("test data num:", len(test_data))
        print("valid metric:", get_valid_metric_name(args))

    collator = Collator(args, tokenizer)
    config.codebook_path = args.codebook_path
    config.alpha = args.alpha
    config.bos_queries = args.bos_queries
    config.decor_rq_codebook_size = 256
    config.decor_rq_n_codebooks = 3
    config.decor_collision_size = 256
    config.decor_latent_size = config.d_model

    model = DECOR.from_pretrained(
        args.ckpt_path,
        config=config,
        tokenizer=tokenizer,
        model_temperature=args.temperature,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = True

    if args.results_file in ("", "./results/test-ddp.json"):
        args.results_file = os.path.join(args.ckpt_path, "test_results_eval_only.json")

    run_generation_eval(args, model, tokenizer, test_data, device, results_file=args.results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DECOR_eval_only')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    evaluate_only(args)
