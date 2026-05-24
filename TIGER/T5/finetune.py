import argparse
import os
import sys
import re
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling_tiger import TIGER
from test import run_generation_eval
# import wandb
from utils import *
from collator import Collator


def _parse_rq_token(token):
    matched = re.fullmatch(r"<([^<>]+)>", token)
    if matched is None:
        return None

    inner = matched.group(1)
    if "_" not in inner:
        return None

    stage_tag, code_tag = inner.rsplit("_", 1)
    if not code_tag.isdigit():
        return None

    stage_tag = stage_tag.lower()
    if len(stage_tag) == 1 and "a" <= stage_tag <= "z":
        stage_idx = ord(stage_tag) - ord("a")
    elif stage_tag.isdigit():
        stage_idx = int(stage_tag)
    else:
        matched_stage = re.fullmatch(r"(?:s|stage)(\d+)", stage_tag)
        if matched_stage is None:
            return None
        stage_idx = int(matched_stage.group(1))

    return stage_idx, int(code_tag)


def init_t5_embeddings_from_rq_codebook(model, tokenizer, new_tokens,codebook_path=""):
    codebook_path = os.path.abspath(codebook_path)
    if not os.path.exists(codebook_path):
        print(f"Skip RQ codebook init: file not found at {codebook_path}")
        return

    saved = np.load(codebook_path, allow_pickle=True)
    stage_codebooks = [np.asarray(codebook, dtype=np.float32) for codebook in saved["codebooks"].tolist()]

    embed_weight = model.get_input_embeddings().weight
    target_dim = embed_weight.shape[1]

    initialized = 0
    skipped = 0

    with torch.no_grad():
        for token in new_tokens:
            parsed = _parse_rq_token(token)
            if parsed is None:
                skipped += 1
                continue

            token_id = tokenizer.convert_tokens_to_ids(token)
            tokenized = tokenizer(token, add_special_tokens=False)["input_ids"]
            if token_id is None or token_id < 0 or len(tokenized) != 1 or tokenized[0] != token_id:
                skipped += 1
                continue

            stage_idx, code_idx = parsed
            if stage_idx >= len(stage_codebooks):
                skipped += 1
                continue

            codebook = stage_codebooks[stage_idx]
            if code_idx >= codebook.shape[0]:
                skipped += 1
                continue

            if codebook.shape[1] < target_dim:
                raise ValueError(
                    f"RQ codebook dim {codebook.shape[1]} is smaller than model dim {target_dim}."
                )

            code_vec = torch.as_tensor(
                codebook[code_idx, :target_dim],
                dtype=embed_weight.dtype,
                device=embed_weight.device,
            )
            embed_weight[token_id].copy_(code_vec)
            initialized += 1
    print("----------------------------------------------------INIT----------------------------------------")
    print(
        f"Initialized {initialized} token embeddings from RQ codebooks "
        f"(skipped {skipped}, path={codebook_path})"
    )




def get_valid_metric_name(args):
    if args.valid_metric == "loss":
        return "loss"
    if args.valid_metric == "ndcg":
        return f"ndcg@{args.valid_metric_k}"
    if args.valid_metric == "hitrate":
        return f"hitrate@{args.valid_metric_k}"
    raise ValueError(f"Unsupported valid metric: {args.valid_metric}")


class TigerTrainer(transformers.Trainer):
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


def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    train_data, valid_data = load_datasets(args)
    test_data = load_test_dataset(args)
    config = T5Config.from_pretrained(args.base_model)
    tokenizer, add_num = build_tokenizer(
        args,
        model_max_length=args.model_max_length,
        new_tokens=train_data.datasets[0].get_new_tokens(),
    )
    pretokenize_datasets(tokenizer, train_data, valid_data, test_data)
    config.mask_token_id = -1
    config.maskgit_target_length = 4
    args.deepspeed = None
    gradient_checkpointing= False
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        print("test data num:", len(test_data))
        print("valid metric:", get_valid_metric_name(args))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(valid_data[100])


    collator = Collator(args, tokenizer)
    model = TIGER(
        config,
        model_temperature=args.temperature,
    )

    model.resize_token_embeddings(len(tokenizer))

    if args.use_codebook_init:
        init_t5_embeddings_from_rq_codebook(model, tokenizer, train_data.datasets[0].get_new_tokens(),args.codebook_path)

    model.to(device)
    if local_rank == 0:
        print(model)

    metric_for_best_model = get_valid_metric_name(args)
    greater_is_better = None if args.valid_metric == "loss" else True

    trainer = TigerTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim="adamw_torch",
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
        valid_metric_mode=args.valid_metric,
        valid_metric_name=metric_for_best_model,
        valid_eval_args=args,
        valid_tokenizer=tokenizer,
    )
    model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    if (not args.no_test_after_train) and local_rank == 0:
        if args.results_file in ("", "./results/test-ddp.json"):
            args.results_file = os.path.join(args.output_dir, "test_results.json")
        eval_model = trainer.model
        run_generation_eval(args, eval_model, tokenizer, test_data, device, results_file=args.results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TIGER')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    train(args)
