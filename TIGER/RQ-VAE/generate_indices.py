from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import json
import os
from collections import defaultdict
from torch.utils.data import DataLoader
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str, default="Instruments", help="dataset")
    parser.add_argument("--root_path", type=str, default="../checkpoint/", help="root path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="full checkpoint path")
    parser.add_argument('--alpha', type=str, default='1e-1', help='legacy checkpoint path component')
    parser.add_argument('--epoch', type=int, default=10000, help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='legacy checkpoint path component')
    parser.add_argument('--output_file', type=str, default=None, help='full output json path')
    parser.add_argument("--data_path", type=str, required=True, help="semantic embedding npy path")
    parser.add_argument('--embedding_dim', type=int, default=256, help='truncate raw embeddings to the first N dims before optional PCA')
    parser.add_argument('--pca_dim', type=int, default=None, help='apply PCA after truncation and reduce to this dimension')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256], help='emb num of every vq')
    return parser.parse_args()


def build_collision_aware_indices(raw_codes):
    semantic_id_to_count = defaultdict(int)
    final_codes = []
    for code in raw_codes:
        semantic_key = tuple(code)
        semantic_id_to_count[semantic_key] += 1
        collision_idx = semantic_id_to_count[semantic_key] - 1
        final_codes.append(list(code) + [collision_idx])
    max_collision = max((code[-1] for code in final_codes), default=0)
    if max_collision >= 256:
        raise ValueError(
            f"Collision index exceeds 255 (max={max_collision}). Increase collision capacity."
        )
    return final_codes


def resolve_device(device_str):
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        try:
            requested_index = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
        except ValueError:
            requested_index = 0
        visible_count = torch.cuda.device_count()
        if requested_index >= visible_count:
            if visible_count == 1:
                return torch.device("cuda:0")
            raise RuntimeError(
                f"Invalid CUDA device ordinal: requested {device_str}, visible device count is {visible_count}"
            )
    return torch.device(device_str)


args_setting = parse_args()
dataset = args_setting.dataset
ckpt_path = args_setting.ckpt_path
if ckpt_path is None:
    ckpt_path = args_setting.root_path + f'alpha{args_setting.alpha}-beta{args_setting.beta}/' + args_setting.checkpoint

output_dir = f"./data/{dataset}/"
output_file = f"{dataset}.index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
output_file = os.path.join(output_dir, output_file)
if args_setting.output_file is not None:
    output_file = args_setting.output_file

output_parent = os.path.dirname(output_file)
if output_parent:
    os.makedirs(output_parent, exist_ok=True)

checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = checkpoint['args']
state_dict = checkpoint['state_dict']

device_str = os.environ.get("DEVICE", "cuda:0")
device = resolve_device(device_str)

if not hasattr(args, 'embedding_dim'):
    args.embedding_dim = args_setting.embedding_dim
if not hasattr(args, 'pca_dim'):
    args.pca_dim = args_setting.pca_dim

data = EmbDataset(
    args_setting.data_path,
    embedding_dim=args_setting.embedding_dim,
    pca_dim=args_setting.pca_dim,
)
model = RQVAE(
    in_dim=data.dim,
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=args.dropout_prob,
    bn=args.bn,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    kmeans_init=args.kmeans_init,
    beta=args.beta,
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

data_loader = DataLoader(
    data,
    num_workers=args.num_workers,
    batch_size=64,
    shuffle=False,
    pin_memory=True,
)

raw_codes_with_indices = []
with torch.no_grad():
    for x, indices in data_loader:
        x = x.to(device)
        _, _, x_indices, _ = model(x)
        x_indices = x_indices.detach().cpu().numpy().tolist()
        batch_indices = indices.detach().cpu().numpy().tolist()
        for idx, emb_idx in enumerate(batch_indices):
            raw_codes_with_indices.append((emb_idx, x_indices[idx]))

ordered_item_ids = [item_id for item_id, _ in raw_codes_with_indices]
ordered_raw_codes = [code for _, code in raw_codes_with_indices]
final_codes = build_collision_aware_indices(ordered_raw_codes)

all_indices = {}
for item_id, final_code in zip(ordered_item_ids, final_codes):
    all_indices[str(item_id)] = [f'<{chr(ord("a") + level)}_{j}>' for level, j in enumerate(final_code)]

with open(output_file, "w", encoding="utf-8") as fp:
    json.dump(all_indices, fp, ensure_ascii=False)
print(f"Saved indices to {output_file}")
