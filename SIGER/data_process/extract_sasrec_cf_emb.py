import argparse
from typing import Any, Optional

import numpy as np
import torch


ATTR_CANDIDATES = [
    "item_embeddings",
    "item_embedding",
]

DICT_KEY_CANDIDATES = [
    "item_embeddings.weight",
    "item_embedding.weight",
    "state_dict.item_embeddings.weight",
    "state_dict.item_embedding.weight",
    "model.item_embeddings.weight",
    "model.item_embedding.weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract real-item collaborative embeddings from a SASRec checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="SASRec checkpoint path.")
    parser.add_argument("--output", type=str, required=True, help="Output .pt path.")
    parser.add_argument(
        "--semantic_npy",
        type=str,
        required=True,
        help="Semantic embedding .npy path used to assert the item count.",
    )
    parser.add_argument(
        "--embedding_key",
        type=str,
        default=None,
        help="Optional explicit dotted key path for dict checkpoints, e.g. state_dict.item_embeddings.weight",
    )
    return parser.parse_args()


def resolve_dotted(obj: Any, dotted: str) -> Any:
    current = obj
    for part in dotted.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Missing key '{part}' while resolving '{dotted}'")
            current = current[part]
        else:
            if not hasattr(current, part):
                raise AttributeError(f"Missing attribute '{part}' while resolving '{dotted}'")
            current = getattr(current, part)
    return current


def extract_tensor(source: Any) -> torch.Tensor:
    if isinstance(source, torch.Tensor):
        return source.detach().cpu()
    if isinstance(source, torch.nn.Embedding):
        return source.weight.detach().cpu()
    if hasattr(source, "weight") and isinstance(source.weight, torch.Tensor):
        return source.weight.detach().cpu()
    raise TypeError(f"Unsupported embedding source type: {type(source)}")


def find_embedding_source(checkpoint: Any, explicit_key: Optional[str]) -> Any:
    # 1. 如果指定了 key，直接寻找
    if explicit_key is not None:
        return resolve_dotted(checkpoint, explicit_key)

    # 2. 专门针对 RecBole 的逻辑：如果存在 state_dict，直接去 state_dict 里找
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        # 匹配 item_embedding.weight 或 item_embeddings.weight
        for target_key in ["item_embedding.weight", "item_embeddings.weight"]:
            if target_key in sd:
                print(f"Successfully found '{target_key}' in state_dict.")
                return sd[target_key]

        # 如果还是找不到，尝试模糊匹配（处理带前缀的情况）
        for k in sd.keys():
            if "item_embedding.weight" in k:
                print(f"Found fuzzy match: {k}")
                return sd[k]

    # 3. 原有的通用匹配逻辑
    if isinstance(checkpoint, dict):
        for dotted in DICT_KEY_CANDIDATES:
            try:
                return resolve_dotted(checkpoint, dotted)
            except (KeyError, AttributeError):
                continue

    raise ValueError(
        "Could not find SASRec item embedding in the checkpoint. "
        "Please check the printed keys above and pass --embedding_key explicitly."
    )


def main() -> None:
    args = parse_args()

    semantic_emb = np.load(args.semantic_npy)
    if semantic_emb.ndim != 2:
        raise AssertionError(
            f"Semantic embedding must be a 2D array, but got shape {semantic_emb.shape}"
        )
    num_items = semantic_emb.shape[0]

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    embedding_source = find_embedding_source(checkpoint, args.embedding_key)
    embedding_weight = extract_tensor(embedding_source)

    if embedding_weight.ndim != 2:
        raise AssertionError(
            f"SASRec item embedding must be 2D, but got shape {tuple(embedding_weight.shape)}"
        )

    if embedding_weight.shape[0] == num_items + 1:#如果语义嵌入的行数是num_items+1，说明SASRec的item embedding包含了一个额外的pad项（通常在第0行）。去掉
        cf_emb = embedding_weight[1:].contiguous()
    elif embedding_weight.shape[0] == num_items:
        cf_emb = embedding_weight.contiguous()
    else:
        raise AssertionError(
            "Item count mismatch: semantic npy has "
            f"{num_items} rows, while SASRec item embedding has {embedding_weight.shape[0]} rows. "
            "Expected either num_items or num_items + 1 (with pad at row 0)."
        )

    assert cf_emb.shape[0] == num_items, "Collaborative embedding row count must match semantic embedding row count."

    torch.save(cf_emb, args.output)

    print(f"Semantic embedding shape: {semantic_emb.shape}")
    print(f"Raw SASRec item embedding shape: {tuple(embedding_weight.shape)}")
    print(f"Saved collaborative embedding shape: {tuple(cf_emb.shape)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
