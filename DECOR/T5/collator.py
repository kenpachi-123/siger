import torch


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


def pad_tokenized_batch(sequences, pad_token_id, pad_value=None):
    pad_value = pad_token_id if pad_value is None else pad_value
    width = max(1, max(len(sequence) for sequence in sequences))
    padded = torch.full((len(sequences), width), pad_value, dtype=torch.long)
    for row_idx, sequence in enumerate(sequences):
        if sequence:
            padded[row_idx, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded


class Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        if "attention_mask" in batch[0]:
            inputs = {
                "input_ids": pad_tokenized_batch(
                    [d["input_ids"] for d in batch], self.tokenizer.pad_token_id
                ),
                "attention_mask": pad_tokenized_batch(
                    [d["attention_mask"] for d in batch], self.tokenizer.pad_token_id, 0
                ),
            }
            labels = pad_tokenized_batch(
                [d["label_ids"] for d in batch], self.tokenizer.pad_token_id
            )
            labels[labels == self.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels
            inputs["maskgit_target_ids"] = pad_tokenized_batch(
                [d["maskgit_target_ids"] for d in batch], self.tokenizer.pad_token_id
            )
        else:
            input_texts = [d["input_ids"] for d in batch]
            label_texts = [d["labels"] for d in batch]

            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )

            labels = self.tokenizer(
                label_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            inputs['labels'] = labels['input_ids']
            inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
            maskgit_targets = self.tokenizer(
                label_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            inputs["maskgit_target_ids"] = maskgit_targets["input_ids"]
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
        targets = [d["labels"] for d in batch]
        if "attention_mask" in batch[0]:
            inputs = {
                "input_ids": pad_tokenized_batch(
                    [d["input_ids"] for d in batch], self.tokenizer.pad_token_id
                ),
                "attention_mask": pad_tokenized_batch(
                    [d["attention_mask"] for d in batch], self.tokenizer.pad_token_id, 0
                ),
            }
        else:
            input_texts = [d["input_ids"] for d in batch]
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

        return (inputs, targets)
