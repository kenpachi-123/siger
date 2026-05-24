import argparse
import json
import os
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None


SEQ_CANDIDATES = [
    "item_seq",
    "items",
    "sequence",
    "item_ids",
    "item_id_list",
    "seq",
]

USER_CANDIDATES = [
    "user_id",
    "uid",
    "user",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert split parquet files into LETTER inter.json format."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--test_file", type=str, required=True, help="Test parquet path.")
    parser.add_argument("--train_file", type=str, default=None, help="Optional train parquet path.")
    parser.add_argument("--valid_file", type=str, default=None, help="Optional valid parquet path.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="../data",
        help="Root output directory. Final file goes to output_root/dataset/dataset.inter.json",
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default=None,
        help="Column that stores the item sequence. If omitted, infer from common names.",
    )
    parser.add_argument(
        "--user_col",
        type=str,
        default=None,
        help="Column that stores the user id. If omitted, infer from common names or fall back to row index.",
    )
    parser.add_argument(
        "--item_id_base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Original item id base in parquet. LETTER expects 0-based ids in inter.json.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=3,
        help="Drop users whose full test sequence is shorter than this length.",
    )
    parser.add_argument(
        "--validate_prefix",
        action="store_true",
        help="Validate that train == test[:-2] and valid == test[:-1] for shared users.",
    )
    return parser.parse_args()


def infer_column(df: pd.DataFrame, explicit: Optional[str], candidates: List[str], kind: str) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"{kind} column '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit

    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_sequence(value: Any) -> List[int]:
    if value is None:
        raise ValueError("Sequence value is None.")

    if isinstance(value, list):
        seq = value
    elif isinstance(value, tuple):
        seq = list(value)
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        seq = value.tolist()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "," in text:
                parsed = [part.strip() for part in text.split(",") if part.strip()]
            else:
                parsed = [part.strip() for part in text.split() if part.strip()]
        if isinstance(parsed, list):
            seq = parsed
        else:
            raise ValueError(f"Unsupported string sequence value: {value}")
    else:
        raise TypeError(f"Unsupported sequence type: {type(value)}")

    normalized = []
    for item in seq:
        if item is None or item == "":
            continue
        normalized.append(int(item))
    return normalized


def extract_row_sequence(row: pd.Series, sequence_col: Optional[str]) -> List[int]:
    if sequence_col is not None:
        return normalize_sequence(row[sequence_col])

    if "history" in row and "target" in row:
        history = normalize_sequence(row["history"])
        target = int(row["target"])
        return history + [target]

    raise KeyError(
        "Could not find a usable sequence representation. "
        "Expected either a full sequence column or the pair of columns: history + target."
    )


def dataframe_to_sequences(
    parquet_file: str,
    sequence_col: Optional[str],
    user_col: Optional[str],
    item_id_base: int,
    min_seq_len: int,
) -> Dict[str, List[int]]:
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required to read parquet files. Install `pandas` and a parquet backend "
            "such as `pyarrow`, then rerun this script."
        ) from _PANDAS_IMPORT_ERROR
    df = pd.read_parquet(parquet_file)
    seq_col = infer_column(df, sequence_col, SEQ_CANDIDATES, "sequence")
    # Common sequential-rec parquet format stores prefix in `history` and the next
    # item in `target`; in that case the full sequence is history + [target].
    if seq_col is None and not {"history", "target"}.issubset(df.columns):
        raise KeyError(
            f"Could not infer sequence column from {list(df.columns)}. "
            "Expected either a full sequence column or history/target columns."
        )

    usr_col = infer_column(df, user_col, USER_CANDIDATES, "user")
    sequences: Dict[str, List[int]] = {}

    for idx, row in df.iterrows():
        user_id = row[usr_col] if usr_col is not None else idx
        seq = extract_row_sequence(row, seq_col)
        seq = [item - item_id_base for item in seq]
        if any(item < 0 for item in seq):
            raise ValueError(
                f"Found negative item id after subtracting item_id_base={item_id_base}. "
                "Please check the original id base."
            )
        if len(seq) < min_seq_len:
            continue
        sequences[str(user_id)] = seq

    return sequences


def validate_prefixes(
    full_sequences: Dict[str, List[int]],
    train_sequences: Optional[Dict[str, List[int]]],
    valid_sequences: Optional[Dict[str, List[int]]],
) -> None:
    if train_sequences is not None:
        shared_users = set(full_sequences).intersection(train_sequences)
        for uid in shared_users:
            expected = full_sequences[uid][:-2]
            actual = train_sequences[uid]
            if actual != expected:
                raise AssertionError(
                    f"Train prefix mismatch for user {uid}: expected {expected[-10:]}, got {actual[-10:]}"
                )

    if valid_sequences is not None:
        shared_users = set(full_sequences).intersection(valid_sequences)
        for uid in shared_users:
            expected = full_sequences[uid][:-1]
            actual = valid_sequences[uid]
            if actual != expected:
                raise AssertionError(
                    f"Valid prefix mismatch for user {uid}: expected {expected[-10:]}, got {actual[-10:]}"
                )


def main() -> None:
    args = parse_args()

    test_sequences = dataframe_to_sequences(
        parquet_file=args.test_file,
        sequence_col=args.sequence_col,
        user_col=args.user_col,
        item_id_base=args.item_id_base,
        min_seq_len=args.min_seq_len,
    )

    train_sequences = None
    valid_sequences = None
    if args.train_file is not None:
        train_sequences = dataframe_to_sequences(
            parquet_file=args.train_file,
            sequence_col=args.sequence_col,
            user_col=args.user_col,
            item_id_base=args.item_id_base,
            min_seq_len=0,
        )
    if args.valid_file is not None:
        valid_sequences = dataframe_to_sequences(
            parquet_file=args.valid_file,
            sequence_col=args.sequence_col,
            user_col=args.user_col,
            item_id_base=args.item_id_base,
            min_seq_len=0,
        )

    if args.validate_prefix:
        validate_prefixes(test_sequences, train_sequences, valid_sequences)

    output_dir = os.path.join(args.output_root, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.dataset}.inter.json")

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(test_sequences, fp, ensure_ascii=False)

    lengths = [len(seq) for seq in test_sequences.values()]
    all_items = set()
    for seq in test_sequences.values():
        all_items.update(seq)

    print(f"Saved {len(test_sequences)} users to {output_file}")
    if lengths:
        print(
            f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) / len(lengths):.2f}"
        )
        print(f"Observed item ids: min={min(all_items)}, max={max(all_items)}, unique={len(all_items)}")
    else:
        print("No sequences were saved. Please check the parquet contents and min_seq_len.")


if __name__ == "__main__":
    main()
