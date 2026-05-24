import argparse
import json
import os

import numpy as np
import torch



def config_get(config, key, default):
    if isinstance(config, dict):
        return config.get(key, default)
    if key in config:
        return config[key]
    return default


def load_array_from_ckpt(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["config"], ckpt["state_dict"], ckpt.get("other_parameter", {})


def export_item_embedding(state_dict, output_path):
    item_emb = state_dict["item_embedding.weight"][1:]

    np.save(output_path, item_emb.detach().float().numpy())


def export_side_info_embeddings(config, other_parameter, output_dir):
    layers = other_parameter["feature_embed_layer_list"]
    item_num = layers[0].dataset.item_num
    item_idx = torch.arange(1, item_num, dtype=torch.long).unsqueeze(0)

    for idx, field in enumerate(config["selected_features"]):
        layer = layers[idx].to("cpu")
        layer.eval()
        with torch.no_grad():
            sparse_embedding, _ = layer(None, item_idx)

        emb = sparse_embedding["item"].squeeze(0).squeeze(1).detach().float().numpy()
        np.save(os.path.join(output_dir, f"sideinfo_{field}_embeddings.npy"), emb)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    config, state_dict, other_parameter = load_array_from_ckpt(args.ckpt_path)

    if args.output_path is not None:
        item_output_path = args.output_path
        output_dir = os.path.dirname(item_output_path) or "."
    else:
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.ckpt_path), "extracted_embeddings"
        )
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
        item_output_path = os.path.join(output_dir, f"{ckpt_name}_item_embeddings.npy")

    os.makedirs(output_dir, exist_ok=True)

    export_item_embedding(
        state_dict=state_dict,
        output_path=item_output_path,
    )
    #export_side_info_embeddings(config, other_parameter, output_dir)


if __name__ == "__main__":
    main()
