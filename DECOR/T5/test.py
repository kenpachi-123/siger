import argparse
import json
import os

import torch
from collator import TestCollator
from evaluate import get_metrics_results, get_topk_results
from generation_trie import Trie
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config

from modeling_decor import DECOR
from utils import *


def run_generation_eval(
    args,
    model,
    tokenizer,
    test_data,
    device,
    results_file=None,
    use_trie_override=None,
    metrics_override=None,
    prompt_ids_override=None,
    verbose=True,
):
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    use_trie = not args.no_use_trie
    if use_trie_override is not None:
        use_trie = use_trie_override
    prefix_allowed_tokens = None
    candidate_token_ids = None
    if use_trie:
        candidate_trie = Trie(
            [[0] + tokenizer.encode(candidate) for candidate in all_items]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    eval_model = model.module if hasattr(model, "module") else model
    if getattr(eval_model, "decoder_type", "ar") == "maskgit":
        candidate_token_ids = []
        seen_candidate_token_ids = set()
        if hasattr(test_data, "indices"):
            for index_tokens in test_data.indices.values():
                token_ids = []
                valid_candidate = True
                for token in index_tokens:
                    encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
                    if len(encoded) != 1:
                        valid_candidate = False
                        break
                    token_ids.append(encoded[0])
                token_tuple = tuple(token_ids)
                if (
                    valid_candidate
                    and len(token_ids) == getattr(eval_model, "maskgit_target_length", 4)
                    and token_tuple not in seen_candidate_token_ids
                ):
                    candidate_token_ids.append(token_ids)
                    seen_candidate_token_ids.add(token_tuple)
        else:
            for candidate in all_items:
                token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                token_tuple = tuple(token_ids)
                if (
                    len(token_ids) == getattr(eval_model, "maskgit_target_length", 4)
                    and token_tuple not in seen_candidate_token_ids
                ):
                    candidate_token_ids.append(token_ids)
                    seen_candidate_token_ids.add(token_tuple)
        candidate_token_ids = torch.tensor(candidate_token_ids, dtype=torch.long)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if prompt_ids_override is None:
        prompt_ids = [
            int(_) for _ in str(args.test_prompt_ids).split(",") if str(_).strip()
        ]
    else:
        prompt_ids = list(prompt_ids_override)
    if verbose:
        print("data num:", len(test_data))
        print("use trie:", use_trie)

    was_training = eval_model.training
    original_use_cache = getattr(eval_model.config, "use_cache", None)
    if original_use_cache is not None:
        eval_model.config.use_cache = True
    eval_model.eval()

    metrics = (
        metrics_override.split(",")
        if isinstance(metrics_override, str)
        else args.metrics.split(",")
    )
    all_prompt_results = []
    with torch.inference_mode():
        for prompt_id in prompt_ids:
            if hasattr(test_loader.dataset, "refresh_eval_samples"):
                test_loader.dataset.refresh_eval_samples()
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader, disable=not verbose)):
                inputs = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch[0].items()
                }
                targets = batch[1]
                total += len(targets)
                if verbose and step == 0:
                    print(
                        "first batch shapes:",
                        {
                            "input_ids": tuple(inputs["input_ids"].shape),
                            "attention_mask": tuple(inputs["attention_mask"].shape),
                            "targets": len(targets),
                        },
                    )

                with get_eval_autocast_context(args, device):
                    if getattr(eval_model, "decoder_type", "ar") == "maskgit":
                        output = eval_model.generate_maskgit(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            candidate_token_ids=candidate_token_ids,
                            num_return_sequences=args.num_beams,
                            history_item_ids=inputs.get("history_item_ids"),
                        )
                    else:
                        generate_kwargs = dict(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            num_beams=args.num_beams,
                            num_return_sequences=args.num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                        )
                        if "history_item_ids" in inputs:
                            generate_kwargs["history_item_ids"] = inputs["history_item_ids"]
                        if use_trie:
                            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens
                        output = eval_model.generate(**generate_kwargs)
                output_ids = output["sequences"]
                scores = output["sequences_scores"].detach().cpu().tolist()

                predictions = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(
                    predictions,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                    dedup_predictions=args.dedup_predictions,
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                temp = {}
                for m in metrics_results:
                    temp[m] = metrics_results[m] / total
                if verbose:
                    print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            if verbose:
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")
                print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    if verbose:
        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    if results_file is not None:
        results_dir = os.path.dirname(results_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(save_data, f, indent=4)

    if original_use_cache is not None:
        eval_model.config.use_cache = original_use_cache
    if was_training:
        eval_model.train()

    return save_data


def test(args):
    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)

    tokenizer_source = args.ckpt_path if os.path.isdir(args.ckpt_path) else args.base_model
    config = T5Config.from_pretrained(tokenizer_source, local_files_only=True)

    train_data, valid_data = load_datasets(args)
    tokenizer, add_num = build_tokenizer(
        args,
        model_max_length=config.n_positions if hasattr(config, "n_positions") else 512,
        new_tokens=train_data.datasets[0].get_new_tokens(),
        tokenizer_source=tokenizer_source,
        local_files_only=True,
    )
    pretokenize_datasets(tokenizer, train_data, valid_data)
    config.vocab_size = len(tokenizer)

    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))

    model = DECOR.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    test_data = load_test_dataset(args)
    pretokenize_datasets(tokenizer, test_data)
    run_generation_eval(
        args, model, tokenizer, test_data, device, results_file=args.results_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)
