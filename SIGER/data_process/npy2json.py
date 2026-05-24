import argparse
import json
import os
from typing import List

import numpy as np


DEFAULT_PREFIXES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert integer code npy array into LETTER index json format."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input npy file path.")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output json path. Values will be token lists such as ['<a_1>', '<b_2>', ...].",
    )
    parser.add_argument(
        "--key_base",
        type=int,
        default=0,
        choices=[0, 1],
        help="Start item ids from 0 or 1 in the output json keys.",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=",".join(DEFAULT_PREFIXES),
        help="Comma-separated token prefixes to use for each code position.",
    )
    return parser.parse_args()


def load_codes(path: str) -> np.ndarray:
    codes = np.load(path, allow_pickle=True)
    if not isinstance(codes, np.ndarray):
        raise TypeError(f"Expected numpy array from {path}, got {type(codes)}")
    if codes.ndim != 2:
        raise ValueError(f"Expected 2D array of shape [num_items, code_len], got {codes.shape}")
    return codes


def build_prefixes(prefix_text: str, code_len: int) -> List[str]:
    prefixes = [p.strip() for p in prefix_text.split(",") if p.strip()]
    if len(prefixes) < code_len:
        raise ValueError(
            f"Need at least {code_len} prefixes for code length {code_len}, got {len(prefixes)}"
        )
    return prefixes[:code_len]


def main() -> None:
    args = parse_args()
    codes = load_codes(args.input_file)
    prefixes = build_prefixes(args.prefixes, codes.shape[1])

    indices = {}
    for item_offset, row in enumerate(codes):
        item_id = str(item_offset + args.key_base)
        token_row = [f"<{prefix}_{int(code)}>" for prefix, code in zip(prefixes, row.tolist())]
        indices[item_id] = token_row

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as fp:
        json.dump(indices, fp, ensure_ascii=False)

    print(f"Saved: {args.output_file}")
    print(f"Items: {len(indices)}")
    print(f"Code length: {codes.shape[1]}")
    print(f"Prefixes: {prefixes}")
    sample_keys = list(indices.keys())[:3]
    for key in sample_keys:
        print(f"{key}: {indices[key]}")


if __name__ == "__main__":
    main()
