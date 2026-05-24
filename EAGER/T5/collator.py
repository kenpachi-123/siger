import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist


def build_padded_history_item_ids(batch):
    history_width = max(
        1,
        max(len(sample.get("history_item_ids", [])) for sample in batch),
    )
    history_item_ids = torch.full(
        (len(batch), history_width),
        -1,
        dtype=torch.long,
    )
    for row_idx, sample in enumerate(batch):
        sample_history = [int(item_id) for item_id in sample.get("history_item_ids", [])]
        if sample_history:
            history_item_ids[row_idx, :len(sample_history)] = torch.tensor(
                sample_history,
                dtype=torch.long,
            )
    return history_item_ids


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]
        cf_label_texts = [d.get("cf_labels", d["labels"]) for d in batch]
        sem_label_texts = [d.get("sem_labels", d["labels"]) for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        cf_labels = self.tokenizer(cf_label_texts,
                                   return_tensors="pt",
                                   padding="longest",
                                   max_length=self.tokenizer.model_max_length,
                                   truncation=True,
                                   return_attention_mask=True)
        sem_labels = self.tokenizer(sem_label_texts,
                                    return_tensors="pt",
                                    padding="longest",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        inputs['cf_labels'] = cf_labels['input_ids']
        inputs['cf_labels'][inputs['cf_labels'] == self.tokenizer.pad_token_id] = -100
        inputs['sem_labels'] = sem_labels['input_ids']
        inputs['sem_labels'][inputs['sem_labels'] == self.tokenizer.pad_token_id] = -100
        if "target_item_ids" in batch[0]:
            inputs["target_item_ids"] = torch.tensor(
                [int(d["target_item_ids"]) for d in batch],
                dtype=torch.long,
            )
        if "history_item_ids" in batch[0]:
            inputs["history_item_ids"] = build_padded_history_item_ids(batch)

        return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        target_item_ids = [d.get("target_item_ids") for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        if "history_item_ids" in batch[0]:
            inputs["history_item_ids"] = build_padded_history_item_ids(batch)

        return (inputs, targets, target_item_ids)
