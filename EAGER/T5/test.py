import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import BaseModelOutput
from collator import TestCollator
from evaluate import get_metrics_results, get_topk_results

# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config
from modeling_eager import EAGER
from utils import *


def _encode_single_token(tokenizer, token: str) -> int:
    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected token `{token}` to map to a single tokenizer id, but got {token_ids}.")
    return int(token_ids[0])


def build_branch_token_maps(dataset, tokenizer):
    branch_sources = {
        "cf": dataset.cf_indices,
        "sem": dataset.sem_indices,
    }
    branch_maps = {}
    for branch_name, indices in branch_sources.items():
        item_to_tokens = {}
        first_to_second = {}
        token_length = None
        for item_id_str, code_tokens in indices.items():
            encoded_tokens = tuple(_encode_single_token(tokenizer, token) for token in code_tokens)
            if token_length is None:
                token_length = len(encoded_tokens)
            elif token_length != len(encoded_tokens):
                raise ValueError(f"Inconsistent code length in {branch_name} indices.")
            if token_length != 2:
                raise ValueError(
                    f"Original EAGER rerank path currently expects 2-level codes, got {token_length} in {branch_name}."
                )
            item_id = int(item_id_str)
            item_to_tokens[item_id] = encoded_tokens
            first_to_second.setdefault(encoded_tokens[0], []).append((encoded_tokens[1], item_id))
        branch_maps[branch_name] = {
            "item_to_tokens": item_to_tokens,
            "first_token_ids": torch.tensor(sorted(first_to_second.keys()), dtype=torch.long),
            "first_to_second": {
                first_token: sorted(second_items, key=lambda pair: pair[0])
                for first_token, second_items in first_to_second.items()
            },
        }
    return branch_maps


def _select_branch_decoder(model, branch_name: str):
    return model.decoder_cf if branch_name == "cf" else model.decoder_sem


def _gather_token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    token_log_probs = F.log_softmax(logits, dim=-1)
    gathered = token_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=-1)


def _score_branch_candidates(
    model,
    branch_name: str,
    encoder_outputs: BaseModelOutput,
    encoder_attention_mask: torch.Tensor,
    candidate_owner_indices: torch.Tensor,
    candidate_item_ids: torch.Tensor,
    branch_maps,
):
    if candidate_item_ids.numel() == 0:
        return candidate_item_ids.new_zeros((0,), dtype=torch.float32)

    device = encoder_attention_mask.device
    owner_indices = candidate_owner_indices.to(device=device, dtype=torch.long)
    candidate_item_ids = candidate_item_ids.to(device=device, dtype=torch.long)
    candidate_label_ids = torch.tensor(
        [branch_maps[branch_name]["item_to_tokens"][int(item_id)] for item_id in candidate_item_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    expanded_encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_outputs.last_hidden_state.index_select(0, owner_indices)
    )
    expanded_attention_mask = encoder_attention_mask.index_select(0, owner_indices)
    decoder = _select_branch_decoder(model, branch_name)
    _, branch_logits = model._decode_branch(
        decoder,
        expanded_encoder_outputs,
        expanded_attention_mask,
        candidate_label_ids,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return _gather_token_log_probs(branch_logits, candidate_label_ids)


def _generate_branch_candidates(
    model,
    branch_name: str,
    encoder_outputs: BaseModelOutput,
    encoder_attention_mask: torch.Tensor,
    branch_maps,
    num_beams: int,
):
    device = encoder_attention_mask.device
    batch_size = encoder_attention_mask.size(0)
    first_token_ids = branch_maps[branch_name]["first_token_ids"].to(device)
    if first_token_ids.numel() == 0:
        return [[] for _ in range(batch_size)]

    decoder = _select_branch_decoder(model, branch_name)
    start_token_id = model.config.decoder_start_token_id
    start_decoder_input_ids = torch.full(
        (batch_size, 1),
        start_token_id,
        dtype=torch.long,
        device=device,
    )
    _, first_logits = model._decode_branch(
        decoder,
        encoder_outputs,
        encoder_attention_mask,
        labels=None,
        decoder_input_ids=start_decoder_input_ids,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    first_log_probs = F.log_softmax(first_logits[:, -1, :], dim=-1)
    candidate_count = min(num_beams, first_token_ids.numel())
    selected_first_scores, selected_first_indices = first_log_probs[:, first_token_ids].topk(candidate_count, dim=-1)
    selected_first_token_ids = first_token_ids[selected_first_indices]

    repeated_encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_outputs.last_hidden_state.repeat_interleave(candidate_count, dim=0)
    )
    repeated_attention_mask = encoder_attention_mask.repeat_interleave(candidate_count, dim=0)
    second_decoder_input_ids = torch.stack(
        [
            torch.full(
                (batch_size * candidate_count,),
                start_token_id,
                dtype=torch.long,
                device=device,
            ),
            selected_first_token_ids.reshape(-1),
        ],
        dim=1,
    )
    _, second_logits = model._decode_branch(
        decoder,
        repeated_encoder_outputs,
        repeated_attention_mask,
        labels=None,
        decoder_input_ids=second_decoder_input_ids,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    second_log_probs = F.log_softmax(second_logits[:, -1, :], dim=-1).view(batch_size, candidate_count, -1)

    branch_candidates = []
    for batch_idx in range(batch_size):
        scored_items = []
        for beam_idx in range(candidate_count):
            first_token_id = int(selected_first_token_ids[batch_idx, beam_idx].item())
            first_score = float(selected_first_scores[batch_idx, beam_idx].item())
            second_candidates = branch_maps[branch_name]["first_to_second"].get(first_token_id, [])
            if not second_candidates:
                continue
            second_token_ids = torch.tensor(
                [second_token for second_token, _ in second_candidates],
                dtype=torch.long,
                device=device,
            )
            second_scores = second_log_probs[batch_idx, beam_idx].index_select(0, second_token_ids)
            for (second_token_id, item_id), second_score in zip(second_candidates, second_scores.tolist()):
                scored_items.append((item_id, first_score + float(second_score)))
        scored_items.sort(key=lambda pair: pair[1], reverse=True)
        seen_items = set()
        top_items = []
        for item_id, _ in scored_items:
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            top_items.append(item_id)
            if len(top_items) >= num_beams:
                break
        branch_candidates.append(top_items)
    return branch_candidates


def _rerank_item_id_candidates(model, inputs, branch_maps, num_beams: int):
    branch_encodings = model.encode_branches(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        history_item_ids=inputs.get("history_item_ids"),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    cf_candidates = _generate_branch_candidates(
        model,
        "cf",
        branch_encodings["cf"][0],
        branch_encodings["cf"][1],
        branch_maps,
        num_beams,
    )
    sem_candidates = _generate_branch_candidates(
        model,
        "sem",
        branch_encodings["sem"][0],
        branch_encodings["sem"][1],
        branch_maps,
        num_beams,
    )

    batch_predictions = []
    for batch_idx in range(len(cf_candidates)):
        merged_candidates = []
        seen_item_ids = set()
        for item_id in cf_candidates[batch_idx] + sem_candidates[batch_idx]:
            if item_id not in seen_item_ids:
                seen_item_ids.add(item_id)
                merged_candidates.append(item_id)
        batch_predictions.append(merged_candidates[: num_beams * 2])

    flat_owner_indices = []
    flat_candidate_item_ids = []
    for batch_idx, candidate_item_ids in enumerate(batch_predictions):
        flat_owner_indices.extend([batch_idx] * len(candidate_item_ids))
        flat_candidate_item_ids.extend(candidate_item_ids)

    if not flat_candidate_item_ids:
        return [[] for _ in range(inputs["input_ids"].size(0))]

    device = inputs["input_ids"].device
    owner_index_tensor = torch.tensor(flat_owner_indices, dtype=torch.long, device=device)
    candidate_item_tensor = torch.tensor(flat_candidate_item_ids, dtype=torch.long, device=device)

    cf_scores = _score_branch_candidates(
        model,
        "cf",
        branch_encodings["cf"][0],
        branch_encodings["cf"][1],
        owner_index_tensor,
        candidate_item_tensor,
        branch_maps,
    )
    sem_scores = _score_branch_candidates(
        model,
        "sem",
        branch_encodings["sem"][0],
        branch_encodings["sem"][1],
        owner_index_tensor,
        candidate_item_tensor,
        branch_maps,
    )
    total_scores = (cf_scores + sem_scores).detach().cpu().tolist()

    ranked_predictions = []
    offset = 0
    for candidate_item_ids in batch_predictions:
        candidate_count = len(candidate_item_ids)
        scored_candidates = list(
            zip(candidate_item_ids, total_scores[offset : offset + candidate_count])
        )
        offset += candidate_count
        scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
        ranked_predictions.append([item_id for item_id, _ in scored_candidates[:num_beams]])
    return ranked_predictions


def _build_item_id_topk_results(batch_predictions, target_item_ids):
    results = []
    for predictions, target_item_id in zip(batch_predictions, target_item_ids):
        target = int(target_item_id)
        results.append([1 if int(prediction) == target else 0 for prediction in predictions])
    return results


def run_generation_eval(
    args,
    model,
    tokenizer,
    test_data,
    device,
    results_file=None,
    metrics_override=None,
    prompt_ids_override=None,
    verbose=True,
):
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()
    eval_model = model.module if hasattr(model, "module") else model
    branch_maps = None
    use_original_item_id_eval = (
        getattr(eval_model, "encoder_input_mode", None) == "item_id"
    )
    if use_original_item_id_eval:
        branch_maps = build_branch_token_maps(test_data, tokenizer)

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
    # with torch.no_grad():
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
                target_item_ids = batch[2]
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

                if use_original_item_id_eval:
                    with get_eval_autocast_context(args, device):
                        ranked_predictions = _rerank_item_id_candidates(
                            eval_model,
                            inputs,
                            branch_maps,
                            args.num_beams,
                        )
                    topk_res = _build_item_id_topk_results(
                        ranked_predictions,
                        target_item_ids,
                    )
                else:
                    with get_eval_autocast_context(args, device):
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

    train_data, valid_data = load_datasets(args)
    test_data = load_test_dataset(args)
    tokenizer_source = (
        args.ckpt_path if os.path.isdir(args.ckpt_path) else args.base_model
    )
    config = T5Config.from_pretrained(tokenizer_source, local_files_only=True)
    tokenizer, add_num = build_tokenizer(
        args,
        model_max_length=512,
        new_tokens=test_data.get_new_tokens(),
        tokenizer_source=tokenizer_source,
        local_files_only=True,
    )
    config.vocab_size = len(tokenizer)

    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))

    model = EAGER.from_pretrained(
        args.ckpt_path,
        config=config,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    run_generation_eval(
        args, model, tokenizer, test_data, device, results_file=args.results_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)
