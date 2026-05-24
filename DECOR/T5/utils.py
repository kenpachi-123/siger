import json
import logging
import os
import random
import datetime
from contextlib import nullcontext

import numpy as np
import torch
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast, T5Tokenizer
from data import SeqRecDataset

def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="./ckpt/TIGER",help="basic model path")

    parser.add_argument("--output_dir", type=str, default="./ckpt",
                        help="The output directory")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="../data",
                        help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".llamaindex-sk4-sk.json", help="the item indices file")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")#1. <a_...><b_...>2. <a_...><b_...> 添加1. prefix
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")#只训练模型输出的部分，一般是sft？，这里是

    parser.add_argument("--train_prompt_sample_num", type=str, default="1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="-1",
                        help="the number of sampling prompts for each task")

    # arguments related for evaluation
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")
    parser.add_argument("--remove_t5_extra_tokens", action="store_true", default=False,
                        help="use a minimal tokenizer that keeps only pad/eos/unk plus sid tokens")

    return parser

def parse_train_args(parser):

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the fallback Trainer optimizer')
    parser.add_argument("--optimizer_impl", type=str, default="adam",
                        choices=["adam", "adamw_torch"],
                        help="optimizer implementation used for training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_steps", type=float, default=10000)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no_test_after_train", action="store_true", default=False)
    parser.add_argument("--valid_metric", type=str, default="loss",
                        choices=["loss", "ndcg", "hitrate"],
                        help="validation metric used for model selection")
    parser.add_argument("--valid_metric_k", type=int, default=10,
                        help="top-k used when valid_metric is ndcg or hitrate")
    parser.add_argument("--valid_sampling_mode", type=str, default="fixed",
                        choices=["fixed", "realtime"],
                        help="fixed samples validation data once; realtime resamples each evaluation")
    parser.add_argument("--valid_sample_ratio", type=float, default=1.0,
                        help="validation sampling ratio in (0, 1]; applies to fixed or realtime sampling")

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--bos_queries", type=int, default=16)
    parser.add_argument("--codebook_path", type=str, default="")
    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="./ckpt",
                        help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=True,
                        help="whether filter illegal items")
    parser.add_argument(
        "--no_use_trie",
        action="store_true",
        default=False,
        help="disable trie-constrained decoding during generation",
    )
    parser.add_argument("--dedup_predictions", action="store_true", default=False,
                        help="deduplicate repeated beam outputs before computing metrics")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")


    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def get_t5_tokenizer_kwargs(args, model_max_length):
    kwargs = {"model_max_length": model_max_length}
    if getattr(args, "remove_t5_extra_tokens", False):
        kwargs["extra_ids"] = 0
    return kwargs


def build_minimal_tokenizer(new_tokens, model_max_length):
    base_tokens = ["<pad>", "</s>", "<unk>"]
    vocab = {}
    for token in base_tokens + sorted(set(new_tokens)):
        if token not in vocab:
            vocab[token] = len(vocab)

    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = Split(Regex(r"<[^>]+>|[^<>\s]+"), behavior="isolated")
    tokenizer_obj.post_processor = TemplateProcessing(
        single="$A </s>",
        special_tokens=[("</s>", vocab["</s>"])],
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        model_max_length=model_max_length,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def build_tokenizer(
    args,
    model_max_length,
    new_tokens=None,
    tokenizer_source=None,
    local_files_only=False,
):
    if getattr(args, "remove_t5_extra_tokens", False):
        if new_tokens is None:
            raise ValueError("`new_tokens` is required when `remove_t5_extra_tokens` is enabled.")
        tokenizer = build_minimal_tokenizer(new_tokens, model_max_length)
        return tokenizer, len(sorted(set(new_tokens)))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source or args.base_model,
        **get_t5_tokenizer_kwargs(args, model_max_length=model_max_length),
        local_files_only=local_files_only,
        use_fast=True,
    )
    add_num = 0
    if new_tokens is not None:
        add_num = tokenizer.add_tokens(new_tokens)
    return tokenizer, add_num


def get_eval_autocast_context(args, device):
    if getattr(device, "type", None) != "cuda":
        return nullcontext()
    if getattr(args, "bf16", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if getattr(args, "fp16", False):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args,"valid",args.valid_prompt_sample_num)

    return train_data, valid_data

def load_test_dataset(args):

    if args.test_task.lower() == "seqrec":
        # test_data = SeqRecDataset(args, mode="test_ranking", sample_num=args.sample_num)
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

def pretokenize_datasets(tokenizer, *datasets):
    for dataset in datasets:
        if hasattr(dataset, "datasets"):
            for child_dataset in dataset.datasets:
                if hasattr(child_dataset, "pretokenize"):
                    child_dataset.pretokenize(tokenizer)
        elif hasattr(dataset, "pretokenize"):
            dataset.pretokenize(tokenizer)


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
