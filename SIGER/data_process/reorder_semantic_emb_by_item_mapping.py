import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorder semantic embeddings to match process_data_t5 item_mapping.npy order."
    )
    parser.add_argument("--semantic_emb", type=str, required=True, help="Original semantic embeddings .npy")
    parser.add_argument("--semantic_item_ids", type=str, required=True, help="Item ids saved alongside semantic embeddings, e.g. asins.npy")
    parser.add_argument("--item_mapping", type=str, required=True, help="item_mapping.npy from process_data_t5.py")
    parser.add_argument("--output", type=str, required=True, help="Reordered semantic embeddings .npy")
    parser.add_argument("--report", type=str, default=None, help="Optional json report path")
    return parser.parse_args()


def main():
    args = parse_args()

    semantic_emb = np.load(args.semantic_emb)
    semantic_item_ids = np.load(args.semantic_item_ids, allow_pickle=True)
    item_mapping = np.load(args.item_mapping, allow_pickle=True).item()

    assert semantic_emb.ndim == 2, f"semantic_emb must be 2D, got {semantic_emb.shape}"
    assert len(semantic_item_ids) == semantic_emb.shape[0], (
        f"semantic_item_ids length {len(semantic_item_ids)} != semantic_emb rows {semantic_emb.shape[0]}"
    )
    assert isinstance(item_mapping, dict), "item_mapping.npy must contain a dict"

    semantic_id_to_row = {str(item_id): idx for idx, item_id in enumerate(semantic_item_ids.tolist())}

    reordered = np.zeros((len(item_mapping), semantic_emb.shape[1]), dtype=semantic_emb.dtype)
    missing_ids = []

    for raw_item_id, mapped_item_id in item_mapping.items():
        raw_item_id = str(raw_item_id)
        if raw_item_id not in semantic_id_to_row:
            missing_ids.append(raw_item_id)
            continue

        target_row = int(mapped_item_id) - 1
        if target_row < 0 or target_row >= len(item_mapping):
            raise AssertionError(f"Invalid mapped item id {mapped_item_id} for raw item {raw_item_id}")

        source_row = semantic_id_to_row[raw_item_id]
        reordered[target_row] = semantic_emb[source_row]

    if missing_ids:
        raise AssertionError(
            f"{len(missing_ids)} item ids from item_mapping are missing in semantic_item_ids. "
            f"Examples: {missing_ids[:10]}"
        )

    np.save(args.output, reordered)
    print(f"Saved reordered semantic embeddings: {args.output}")
    print(f"Shape: {reordered.shape}")

    if args.report is not None:
        report = {
            "num_items": len(item_mapping),
            "embedding_dim": int(reordered.shape[1]),
            "missing_ids": len(missing_ids),
            "output": str(Path(args.output)),
        }
        with open(args.report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
