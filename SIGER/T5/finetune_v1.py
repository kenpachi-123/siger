import argparse
import math
import os
import sys
import re
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling_letter_contra2 import LETTER
from test import run_generation_eval
# import wandb
from utils_v1 import *
from collator import Collator


#RQ_CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "rq_codebooks.npz")
#RQ_CODEBOOK_PATH="/data/pairman/zzw/upload/DIFF/saved_code/Amazon_Beauty/DIFF_Amazon_Beauty_rq_codebooks.npz"

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
        return f"hit@{args.valid_metric_k}"
    raise ValueError(f"Unsupported valid metric: {args.valid_metric}")


def collect_a_code_token_ids(train_data, tokenizer):
    datasets = getattr(train_data, "datasets", [train_data])
    a_code_tokens = set()
    for ds in datasets:
        indices = getattr(ds, "indices", None)
        if not indices:
            continue
        for code_tokens in indices.values():
            if code_tokens:
                a_code_tokens.add(code_tokens[0])
    a_code_token_ids = []
    for token in sorted(a_code_tokens):
        token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(token_ids) == 1:
            a_code_token_ids.append(int(token_ids[0]))
    return sorted(set(a_code_token_ids))


def _collect_item_id_range(dataset):
    datasets = getattr(dataset, "datasets", [dataset])
    mins = []
    maxs = []
    for ds in datasets:
        if not hasattr(ds, "get_item_id_range"):
            continue
        item_min, item_max = ds.get_item_id_range()
        if item_min is None or item_max is None:
            continue
        mins.append(item_min)
        maxs.append(item_max)
    if not mins:
        return None, None
    return min(mins), max(maxs)


def infer_item_id_base(args, model, train_data, valid_data, test_data):
    if args.item_id_base != "auto":
        return int(args.item_id_base)

    item_ranges = [
        _collect_item_id_range(train_data),
        _collect_item_id_range(valid_data),
        _collect_item_id_range(test_data),
    ]
    mins = [item_min for item_min, _ in item_ranges if item_min is not None]
    maxs = [item_max for _, item_max in item_ranges if item_max is not None]
    if not mins or not maxs:
        return 0

    observed_min = min(mins)
    observed_max = max(maxs)
    if observed_min == 0 and observed_max <= model.item_num - 1:
        return 0
    if observed_min >= 1 and observed_max <= model.item_num:
        return 1
    if observed_max == model.item_num - 1:
        return 0
    if observed_max == model.item_num:
        return 1

    print(
        "Warning: unable to infer item_id_base unambiguously. "
        f"Observed item ids range [{observed_min}, {observed_max}], embedding rows={model.item_num}. "
        "Falling back to 0. Set --item_id_base explicitly if needed."
    )
    return 0


class LetterTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        optimizer_impl="adamw_torch",
        valid_metric_mode="loss",
        valid_metric_name="loss",
        valid_eval_args=None,
        valid_tokenizer=None,
        cls_max_weight=None,
        cls_warmup_steps=0,
        cls_weight_scheduler="constant",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_impl = optimizer_impl
        self.valid_metric_mode = valid_metric_mode
        self.valid_metric_name = valid_metric_name
        self.valid_eval_args = valid_eval_args
        self.valid_tokenizer = valid_tokenizer
        self._last_loss_breakdown_logged_step = -1
        self._cls_max_weight = cls_max_weight
        self._cls_warmup_steps = cls_warmup_steps
        self._cls_weight_scheduler = cls_weight_scheduler

    def _get_cls_contra_weight(self, step: int) -> float:
        """Return the scheduled cls_contra_weight at the given training step.

        Schedulers:
          constant      — always returns cls_max_weight (no change).
          linear_warmup — linearly increases from 0 to cls_max_weight over
                          cls_warmup_steps, then stays constant.
          cosine_decay  — cosine-decays from cls_max_weight to 0 over the
                          full training duration.
          warmup_cosine — linear warm-up then cosine decay to 0.
          
          constant_cosine - 1 -> cosine_decay
        """
        max_w = self._cls_max_weight
        if max_w is None or max_w == 0.0:
            return 0.0

        warmup = max(self._cls_warmup_steps, 0)
        total = max(int(self.state.max_steps), 1)
        sched = self._cls_weight_scheduler

        if sched == "constant":
            return max_w

        if sched == "linear_warmup":
            if warmup > 0 and step < warmup:
                return max_w * step / warmup
            return max_w

        if sched == "cosine_decay":
            progress = min(step / total, 1.0)
            return max_w * 0.5 * (1.0 + math.cos(math.pi * progress))

        if sched == "warmup_cosine":
            if warmup > 0 and step < warmup:
                return max_w * step / warmup
            decay_steps = max(total - warmup, 1)
            progress = min((step - warmup) / decay_steps, 1.0)
            return max_w * 0.5 * (1.0 + math.cos(math.pi * progress))

        if sched == "constant_cosine":
            if warmup > 0 and step < warmup:
                return max_w
            decay_steps = max(total - warmup, 1)
            progress = min((step - warmup) / decay_steps, 1.0)
            return max_w * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max_w

    # def create_optimizer(self):
    #     if self.optimizer is not None:
    #         return self.optimizer
    #     if self.optimizer_impl == "adam":
    #         self.optimizer = torch.optim.Adam(
    #             self.model.parameters(),
    #             lr=self.args.learning_rate,
    #             betas=(self.args.adam_beta1, self.args.adam_beta2),
    #             eps=self.args.adam_epsilon,
    #         )
    #         return self.optimizer
    #     return super().create_optimizer()

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
        current_step = int(self.state.global_step)

        # Dynamically update cls_contra_weight according to the schedule.
        if (
            model.training
            and self._cls_max_weight is not None
            and hasattr(model, "use_cls_contra")
            and model.use_cls_contra
        ):
            scheduled_weight = self._get_cls_contra_weight(current_step)
            model.cls_contra_weight = scheduled_weight

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        ranking_loss = None
        if hasattr(model, "latest_loss_breakdown") and model.latest_loss_breakdown:
            ranking_loss = model.latest_loss_breakdown.get("ranking_loss")

        if model.training:
            logging_steps = max(int(getattr(self.args, "logging_steps", 0) or 0), 1)
            if (
                current_step != self._last_loss_breakdown_logged_step
                and current_step % logging_steps == 0
                and hasattr(model, "latest_loss_breakdown")
                and model.latest_loss_breakdown
            ):
                breakdown = {}
                for key, value in model.latest_loss_breakdown.items():
                    if torch.is_tensor(value):
                        breakdown[key] = float(value.detach().item())
                # Also log the current scheduled weight so it's visible in the curve.
                if self._cls_max_weight is not None and hasattr(model, "cls_contra_weight"):
                    breakdown["cls_contra_weight"] = float(model.cls_contra_weight)
                if breakdown:
                    self.log(breakdown)
                    self._last_loss_breakdown_logged_step = current_step
        elif ranking_loss is not None:
            loss = ranking_loss

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
    config.use_cls_contra = args.use_cls_contra
    config.cls_contra_weight = args.cls_contra_weight
    config.cls_temperature = args.cls_temperature
    config.emb_file = args.emb_file
    tokenizer, add_num = build_tokenizer(
        args,
        model_max_length=args.model_max_length,
        new_tokens=train_data.datasets[0].get_new_tokens(),
    )
    config.a_code_token_ids = collect_a_code_token_ids(train_data, tokenizer)
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
    model = LETTER(
        config,
        model_temperature=args.temperature,
        use_cls_contra=args.use_cls_contra,
        cls_contra_weight=args.cls_contra_weight,
        cls_contra_type=args.cls_contra_type,
        cls_temperature=args.cls_temperature,
        item_id_base=0,
        emb_file=args.emb_file,
        cls_num_negatives=args.cls_num_negatives,
    )
    if args.use_cls_contra:
        item_id_base = infer_item_id_base(args, model, train_data, valid_data, test_data)
        model.set_hyper(item_id_base=item_id_base)
        model.config.item_id_base = item_id_base
        if local_rank == 0:
            train_min, train_max = _collect_item_id_range(train_data)
            valid_min, valid_max = _collect_item_id_range(valid_data)
            test_min, test_max = _collect_item_id_range(test_data)
            print(
                "CLS contrastive item-id stats:",
                {
                    "train_range": [train_min, train_max],
                    "valid_range": [valid_min, valid_max],
                    "test_range": [test_min, test_max],
                    "embedding_rows": model.item_num,
                    "item_id_base": item_id_base,
                    "cls_temperature": args.cls_temperature,
                    "cls_contra_weight": args.cls_contra_weight,
                },
            )

    model.resize_token_embeddings(len(tokenizer))

    #*********important*********"""added init"""
    if args.use_codebook_init:
        init_t5_embeddings_from_rq_codebook(model, tokenizer, train_data.datasets[0].get_new_tokens(),args.codebook_path)
    #*********important*********"""added init"""

    model.to(device)
    if local_rank == 0:
        print(model)


    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True


    metric_for_best_model = get_valid_metric_name(args)
    greater_is_better = None if args.valid_metric == "loss" else True
    trainer_optim = "adamw_torch"

    trainer = LetterTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        cls_max_weight=args.cls_contra_weight if args.use_cls_contra else None,
        cls_warmup_steps=args.cls_warmup_steps,
        cls_weight_scheduler=args.cls_weight_scheduler,
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
            lr_scheduler_type=args.lr_scheduler_type,
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
        #optimizer_impl=args.optimizer_impl,
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
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    train(args)
