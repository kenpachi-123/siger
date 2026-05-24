import argparse
import json
import os

import numpy as np
import torch

from Tree_Model import Tree


K = 256
INIT_WAY = "embkm"
MAX_ITERS = 100
FEATURE_RATIO = 1.0
PARALL = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Build EAGER quantization codes and export TIGER index json.")
    parser.add_argument("--semantic_emb_path", type=str, required=True)
    parser.add_argument("--collaborative_emb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--item_id_offset", type=int, default=0, choices=[0, 1])
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--tree_name", type=str, required=True, choices=["collaborative", "semantic"])
    return parser.parse_args()


def load_embedding(path):
    if path.endswith(".npy"):
        data = np.load(path)
        tensor = torch.from_numpy(data)
    elif path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            raise TypeError(f"Expected a tensor/array at {path}, but got a dict.")
        tensor = torch.as_tensor(data)
    else:
        raise ValueError(f"Unsupported embedding file: {path}")
    return tensor.float().cpu().contiguous()


def truncate_semantic_embedding(tensor, target_dim=256):
    if tensor.ndim != 2:
        raise ValueError("Semantic embedding must be a 2D tensor.")
    if tensor.shape[1] < target_dim:
        raise ValueError(
            f"Semantic embedding dim {tensor.shape[1]} is smaller than required {target_dim}."
        )
    return tensor[:, :target_dim].contiguous()


def code_to_tokens(code_row):
    tokens = []
    for level, code_value in enumerate(code_row):
        prefix = chr(ord("a") + level) if level < 26 else f"l{level}"
        tokens.append(f"<{prefix}_{int(code_value)}>")
    return tokens


def save_tree_outputs(tree_name, output_dir, k_value, item_id_offset, tree, item_to_code_mat):
    base_name = f"eager_{tree_name}_k{k_value}"
    item_to_code_path = os.path.join(output_dir, f"{base_name}_item_to_code.npy")
    code_to_item_path = os.path.join(output_dir, f"{base_name}_code_to_item.npy")
    index_json_path = os.path.join(output_dir, f"{base_name}.index.json")

    np.save(item_to_code_path, item_to_code_mat.numpy())
    np.save(code_to_item_path, tree.code_to_item.cpu().numpy())

    indices = {}
    for item_idx, code_row in enumerate(item_to_code_mat.tolist()):
        tiger_item_id = item_idx + item_id_offset
        indices[str(tiger_item_id)] = code_to_tokens(code_row)

    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(indices, f, ensure_ascii=True)

    print(f"[{tree_name}] item_to_code -> {item_to_code_path}")
    print(f"[{tree_name}] code_to_item -> {code_to_item_path}")
    print(f"[{tree_name}] tiger index -> {index_json_path}")


def build_one_tree(tree_name, embedding_path, output_dir, cuda_device, item_id_offset, k_value):
    print(f"[{tree_name}] loading embedding from {embedding_path}", flush=True)
    embedding = load_embedding(embedding_path)
    if tree_name == "semantic":
        embedding = truncate_semantic_embedding(embedding)
    item_num = embedding.size(0)
    print(f"[{tree_name}] embedding shape={tuple(embedding.shape)} cuda_device={cuda_device}", flush=True)

    torch.cuda.set_device(cuda_device)
    print(f"[{tree_name}] start Tree construction", flush=True)
    tree = Tree(
        data=embedding,
        max_iters=MAX_ITERS,
        feature_ratio=FEATURE_RATIO,
        item_num=item_num,
        k=k_value,
        init_way=INIT_WAY,
        parall=PARALL,
    )

    item_to_code_mat = torch.full((item_num, tree.tree_height), -1, dtype=torch.int64)
    for item_id, paths in tree.item_to_code.items():
        assert len(paths) > 0
        item_to_code_mat[item_id] = paths[0]

    save_tree_outputs(tree_name, output_dir, k_value, item_id_offset, tree, item_to_code_mat)
    print(f"[{tree_name}] finished tree export", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    emb_path = args.collaborative_emb_path if args.tree_name == "collaborative" else args.semantic_emb_path
    build_one_tree(
        tree_name=args.tree_name,
        embedding_path=emb_path,
        output_dir=args.output_dir,
        cuda_device=args.cuda_device,
        item_id_offset=args.item_id_offset,
        k_value=args.k,
    )


if __name__ == "__main__":
    main()
