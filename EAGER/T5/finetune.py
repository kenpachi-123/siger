import argparse
import os
import sys
from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Config
from modeling_eager import EAGER
from eager_scheduler import InverseSquareRootSchedule
from test import run_generation_eval
# import wandb
from utils import *
from collator import Collator


def _encode_single_token(tokenizer, token: str) -> int:
    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected token `{token}` to map to a single tokenizer id, but got {token_ids}.")
    return int(token_ids[0])


def _build_cf_codebook(cf_indices, tokenizer):
    first_to_second = {}
    for code_tokens in cf_indices.values():
        if len(code_tokens) != 2:
            raise ValueError(
                f"Original EAGER guide losses currently expect 2-level collaborative codes, got {len(code_tokens)}."
            )
        first_token_id = _encode_single_token(tokenizer, code_tokens[0])
        second_token_id = _encode_single_token(tokenizer, code_tokens[1])
        first_to_second.setdefault(first_token_id, set()).add(second_token_id)

    first_token_ids = torch.tensor(sorted(first_to_second.keys()), dtype=torch.long)
    first_to_second_token_ids = {
        first_token_id: torch.tensor(sorted(second_token_ids), dtype=torch.long)
        for first_token_id, second_token_ids in first_to_second.items()
    }
    return first_token_ids, first_to_second_token_ids

def get_valid_metric_name(args):
    if args.valid_metric == "loss":
        return "loss"
    if args.valid_metric == "ndcg":
        return f"ndcg@{args.valid_metric_k}"
    if args.valid_metric == "hitrate":
        return f"hit@{args.valid_metric_k}"
    raise ValueError(f"Unsupported valid metric: {args.valid_metric}")


class EAGERTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        optimizer_impl="adamw_torch",
        scheduler_type="inverse_sqrt",
        scheduler_warmup_updates=2000,
        scheduler_warmup_init_lr=1e-7,
        valid_metric_mode="loss",
        valid_metric_name="loss",
        valid_eval_args=None,
        valid_tokenizer=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_impl = optimizer_impl
        self.scheduler_type = scheduler_type
        self.scheduler_warmup_updates = scheduler_warmup_updates
        self.scheduler_warmup_init_lr = scheduler_warmup_init_lr
        self.valid_metric_mode = valid_metric_mode
        self.valid_metric_name = valid_metric_name
        self.valid_eval_args = valid_eval_args
        self.valid_tokenizer = valid_tokenizer
        self._last_breakdown_log_step = -1

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self.optimizer_impl == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        optimizer = optimizer or self.optimizer
        if self.scheduler_type == "inverse_sqrt":
            self.lr_scheduler = InverseSquareRootSchedule(
                optimizer,
                warmup_updates=self.scheduler_warmup_updates,
                warmup_init_lr=self.scheduler_warmup_init_lr,
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer=optimizer)

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
    # ddp = True
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
    if not args.cf_teacher_emb_path or not args.sem_teacher_emb_path:
        raise ValueError("`cf_teacher_emb_path` and `sem_teacher_emb_path` are required for EAGER training.")

    config.encoder_input_mode = args.encoder_input_mode
    config.cf_teacher_emb_path = args.cf_teacher_emb_path
    config.sem_teacher_emb_path = args.sem_teacher_emb_path
    config.num_layers = args.encoder_num_layers
    config.num_decoder_layers = args.decoder_num_layers
    config.aux_num_layers = args.aux_num_layers
    config.item_id_base = args.item_id_base

    model = EAGER(
        config,
        encoder_input_mode=args.encoder_input_mode,
        cf_teacher_emb_path=args.cf_teacher_emb_path,
        sem_teacher_emb_path=args.sem_teacher_emb_path,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        aux_num_layers=args.aux_num_layers,
        item_id_base=args.item_id_base,
        model_temperature=args.temperature,
    )

    model.resize_token_embeddings(len(tokenizer))
    cf_first_token_ids, cf_first_to_second_token_ids = _build_cf_codebook(
        train_data.datasets[0].cf_indices,
        tokenizer,
    )
    model.set_cf_codebook(cf_first_token_ids, cf_first_to_second_token_ids)

    model.to(device)
    if local_rank == 0:
        print(model)


    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True


    metric_for_best_model = get_valid_metric_name(args)
    greater_is_better = None if args.valid_metric == "loss" else True
    trainer_optim = "adamw_torch"

    trainer = EAGERTrainer(
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
           # max_grad_norm=args.max_grad_norm,
            lr_scheduler_type="constant",
            #fp16=args.fp16,
            bf16=args.bf16,
            #fp16_full_eval=args.fp16,
            #bf16_full_eval=args.bf16,
            logging_steps=args.logging_step,
            optim="adamw_torch",
            # gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_safetensors=False,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=20)],
        optimizer_impl=args.optimizer_impl,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_updates=args.warmup_updates,
        scheduler_warmup_init_lr=args.warmup_init_lr,
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
    if trainer.is_world_process_zero():
        save_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        save_model.save_pretrained(args.output_dir, safe_serialization=False)
        if trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(args.output_dir)

    if (not args.no_test_after_train) and local_rank == 0:
        if args.results_file in ("", "./results/test-ddp.json"):
            args.results_file = os.path.join(args.output_dir, "test_results.json")
        eval_model = trainer.model
        run_generation_eval(args, eval_model, tokenizer, test_data, device, results_file=args.results_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    train(args)
