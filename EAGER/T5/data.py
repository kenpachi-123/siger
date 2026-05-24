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
import json
import numpy as np


def load_concatenated_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    docs = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        docs.append(obj)
        idx = end

    if not docs:
        raise ValueError(f"Empty JSON file: {path}")
    if len(docs) == 1:
        return docs[0]

    merged = {}
    for doc in docs:
        if not isinstance(doc, dict):
            raise TypeError(f"Concatenated JSON in {path} must contain only objects.")
        merged.update(doc)
    return merged


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.cf_index_file = args.cf_index_file or args.index_file
        self.sem_index_file = args.sem_index_file or args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        self.indices = load_concatenated_json(os.path.join(self.data_path, self.dataset + self.index_file))
        self.cf_indices = load_concatenated_json(os.path.join(self.data_path, self.dataset + self.cf_index_file))
        self.sem_indices = load_concatenated_json(os.path.join(self.data_path, self.dataset + self.sem_index_file))

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for source in (self.indices, self.cf_indices, self.sem_indices):
            for index in source.values():
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

        self.inters = load_concatenated_json(os.path.join(self.data_path, self.dataset + ".inter.json"))
        self.indices = load_concatenated_json(os.path.join(self.data_path, self.dataset + self.index_file))
        self.cf_indices = load_concatenated_json(os.path.join(self.data_path, self.dataset + self.cf_index_file))
        self.sem_indices = load_concatenated_json(os.path.join(self.data_path, self.dataset + self.sem_index_file))

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
                one_data["cf_item"] = "".join(self.cf_indices[str(int(item_ids[i]))])
                one_data["sem_item"] = "".join(self.sem_indices[str(int(item_ids[i]))])
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
            one_data["cf_item"] = "".join(self.cf_indices[str(int(item_ids[-2]))])
            one_data["sem_item"] = "".join(self.sem_indices[str(int(item_ids[-2]))])
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

    def refresh_eval_samples(self):
        if self.mode != "valid":
            return
        if self.valid_sampling_mode == "realtime":
            self.inter_data = self._sample_valid_data(with_replacement=True)

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
            one_data["cf_item"] = "".join(self.cf_indices[str(int(item_ids[-1]))])
            one_data["sem_item"] = "".join(self.sem_indices[str(int(item_ids[-1]))])
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
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

        return dict(
            input_ids=d["inters"],
            labels=d["item"],
            cf_labels=d.get("cf_item", d["item"]),
            sem_labels=d.get("sem_item", d["item"]),
            target_item_ids=d.get("item_id"),
            history_item_ids=d.get("history_item_ids", []),
        )

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
