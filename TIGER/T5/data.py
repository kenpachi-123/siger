import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.valid_sample_ratio = float(getattr(args, "valid_sample_ratio", 1.0))
        self.valid_sampling_mode = getattr(args, "valid_sampling_mode", "fixed")
        self.full_inter_data = None
        self.tokenizer = None


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.full_inter_data = self._build_valid_data()
            self.inter_data = self._sample_valid_data(
                with_replacement=self.valid_sampling_mode == "realtime"
            )
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            item_ids = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]
                one_data["item_id"] = int(item_ids[i])
                history = items[:i]
                history_item_ids = [int(item_id) for item_id in item_ids[:i]]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                    history_item_ids = history_item_ids[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                one_data["history_item_ids"] = history_item_ids
                inter_data.append(one_data)

        return inter_data
    
    def _build_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            item_ids = self.inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]
            one_data["item_id"] = int(item_ids[-2])
            history = items[:-2]
            history_item_ids = [int(item_id) for item_id in item_ids[:-2]]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
                history_item_ids = history_item_ids[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            one_data["history_item_ids"] = history_item_ids
            inter_data.append(one_data)

        return inter_data

    def _get_valid_sample_count(self):
        total = len(self.full_inter_data) if self.full_inter_data is not None else 0
        if total == 0:
            return 0
        ratio = min(max(self.valid_sample_ratio, 0.0), 1.0)
        if ratio <= 0:
            raise ValueError("`valid_sample_ratio` must be greater than 0.")
        return max(1, int(round(total * ratio)))

    def _sample_valid_data(self, with_replacement=False):
        if self.full_inter_data is None:
            return []
        if self.valid_sample_ratio >= 1.0:
            return copy.deepcopy(self.full_inter_data)

        sample_count = self._get_valid_sample_count()
        all_inter_idx = np.arange(len(self.full_inter_data))
        sample_idx = np.random.choice(
            all_inter_idx, sample_count, replace=with_replacement
        )
        return np.array(self.full_inter_data, dtype=object)[sample_idx].tolist()

    def _pretokenize_data(self, tokenizer, inter_data):
        if not inter_data:
            return
        encoded_inputs = tokenizer(
            [d["inters"] for d in inter_data],
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        encoded_labels = tokenizer(
            [d["item"] for d in inter_data],
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=False,
        )
        encoded_maskgit_targets = tokenizer(
            [d["item"] for d in inter_data],
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for row, input_ids, attention_mask, labels, maskgit_target_ids in zip(
            inter_data,
            encoded_inputs["input_ids"],
            encoded_inputs["attention_mask"],
            encoded_labels["input_ids"],
            encoded_maskgit_targets["input_ids"],
        ):
            row["tokenized_input_ids"] = input_ids
            row["tokenized_attention_mask"] = attention_mask
            row["tokenized_labels"] = labels
            row["tokenized_maskgit_target_ids"] = maskgit_target_ids

    def pretokenize(self, tokenizer):
        self.tokenizer = tokenizer
        if self.full_inter_data is not None:
            self._pretokenize_data(tokenizer, self.full_inter_data)
            self.inter_data = self._sample_valid_data(
                with_replacement=self.valid_sampling_mode == "realtime"
            )
        else:
            self._pretokenize_data(tokenizer, self.inter_data)

    def refresh_eval_samples(self):
        if self.mode != "valid":
            return
        if self.valid_sampling_mode == "realtime":
            self.inter_data = self._sample_valid_data(with_replacement=True)
            if self.tokenizer is not None and self.inter_data and "tokenized_input_ids" not in self.inter_data[0]:
                self._pretokenize_data(self.tokenizer, self.inter_data)

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            item_ids = self.inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]
            one_data["item_id"] = int(item_ids[-1])
            history = items[:-1]
            history_item_ids = [int(item_id) for item_id in item_ids[:-1]]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
                history_item_ids = history_item_ids[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            one_data["history_item_ids"] = history_item_ids
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       
    

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]

        item = dict(
            input_ids=d.get("tokenized_input_ids", d["inters"]),
            labels=d["item"],
            target_item_ids=d.get("item_id"),
            history_item_ids=d.get("history_item_ids", []),
        )
        if "tokenized_attention_mask" in d:
            item["attention_mask"] = d["tokenized_attention_mask"]
            item["label_ids"] = d["tokenized_labels"]
            item["maskgit_target_ids"] = d["tokenized_maskgit_target_ids"]
        return item

    def get_item_id_range(self):
        mins = []
        maxs = []
        for items in self.inters.values():
            if not items:
                continue
            mins.append(min(items))
            maxs.append(max(items))
        if not mins:
            return None, None
        return min(mins), max(maxs)
