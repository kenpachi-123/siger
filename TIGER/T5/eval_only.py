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

    model = TIGER.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = True

    if args.results_file in ("", "./results/test-ddp.json"):
        args.results_file = os.path.join(args.ckpt_path, "test_results_eval_only.json")

    run_generation_eval(args, model, tokenizer, test_data, device, results_file=args.results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TIGER_eval_only')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    evaluate_only(args)
