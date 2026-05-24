import collections
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
import argparse
import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def resolve_collisions(encoded_x, codes, codebooks):
    """
    Greedily resolve collisions by changing at most one stage code for each
    colliding item, preferring the smallest extra quantization cost.
    """
    new_codes = codes.copy()
    n_samples, n_stages = new_codes.shape

    stage_distances = []
    residual = encoded_x.copy()
    for stage in range(n_stages):
        centers = codebooks[stage]
        dists = (
            np.sum(residual ** 2, axis=1, keepdims=True)
            + np.sum(centers ** 2, axis=1, keepdims=True).T
            - 2 * residual @ centers.T
        )
        stage_distances.append(dists)
        residual = residual - centers[new_codes[:, stage]]

    code_tuples = [tuple(row.tolist()) for row in new_codes]
    combo_counter = collections.Counter(code_tuples)
    collision_combos = {combo for combo, cnt in combo_counter.items() if cnt > 1}

    if not collision_combos:
        print("Collision resolve: no collisions, skip.")
        return new_codes

    collision_groups = collections.defaultdict(list)
    for item_idx, combo in enumerate(code_tuples):
        if combo in collision_combos:
            collision_groups[combo].append(item_idx)

    total_to_reassign = sum(len(group) - 1 for group in collision_groups.values())
    print(
        f"Collision resolve: found {len(collision_groups)} collision groups, "
        f"{total_to_reassign} items need reassignment."
    )

    used_combos = set(code_tuples)
    resolved = 0

    for combo, indices in collision_groups.items():
        current_errs = []
        for idx in indices:
            err = sum(stage_distances[s][idx, new_codes[idx, s]] for s in range(n_stages))
            current_errs.append(err)

        keep_order = [indices[j] for j in np.argsort(current_errs)]

        for idx in keep_order[1:]:
            best_alt = None
            best_extra = float("inf")

            for stage in range(n_stages):
                cur_code = new_codes[idx, stage]
                dists = stage_distances[stage][idx]
                ranking = np.argsort(dists)

                for alt_code in ranking:
                    if int(alt_code) == int(cur_code):
                        continue

                    candidate = new_codes[idx].copy()
                    candidate[stage] = int(alt_code)
                    candidate_tuple = tuple(candidate.tolist())
                    if candidate_tuple in used_combos:
                        continue

                    extra = float(dists[int(alt_code)] - dists[int(cur_code)])
                    if extra < best_extra:
                        best_extra = extra
                        best_alt = (stage, int(alt_code), candidate_tuple)
                    break

            if best_alt is not None:
                stage_pick, code_pick, candidate_tuple = best_alt
                new_codes[idx, stage_pick] = code_pick
                used_combos.add(candidate_tuple)
                resolved += 1

    remaining = len(new_codes) - len(set(tuple(row.tolist()) for row in new_codes))
    print(
        f"Collision resolve: reassigned {resolved}/{total_to_reassign}, "
        f"remaining collisions={remaining}."
    )
    return new_codes

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument("--ckpt_path", type=str, default=None, help='full checkpoint path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')
    parser.add_argument('--output_file', type=str, default=None, help='full output json path')
    parser.add_argument('--resolve_collision', action='store_true', help='greedily resolve remaining collisions')
    parser.add_argument("--data_path", type=str, required=True, help="Semantic embedding npy path")
    parser.set_defaults(resolve_collision=False)
    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
ckpt_path = args_setting.ckpt_path
if ckpt_path is None:
    ckpt_path = args_setting.root_path + f'alpha{args_setting.alpha}-beta{args_setting.beta}/'+args_setting.checkpoint

output_dir = f"./data/{dataset}/"
output_file = f"{dataset}.index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
output_file = os.path.join(output_dir,output_file)
if args_setting.output_file is not None:
    output_file = args_setting.output_file

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]
cf_embedding = state_dict.get("cf_embedding", torch.empty(0, dtype=torch.float32))

device_str = os.environ.get("DEVICE", "cuda:0")
device = torch.device(device_str)

data = EmbDataset(args_setting.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  cf_embedding=cf_embedding,
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
all_code_indices = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]

for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels[str(idx)] = label

all_encoded_x = []
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1]
    d = d.to(device)

    encoded_x = model.encoder(d).detach().cpu().numpy()
    all_encoded_x.append(encoded_x)
    indices = model.get_indices(d, labels,use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        all_code_indices.append(index.copy())
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)
all_code_indices = np.array(all_code_indices, dtype=np.int32)
all_encoded_x = np.concatenate(all_encoded_x, axis=0)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items]
        d = d[0].to(device)
        indices = model.get_indices(d, labels, use_sk=True)

        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            all_code_indices[item] = index.copy()
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1

if args_setting.resolve_collision and not check_collision(all_indices_str):
    codebooks = [layer.embedding.weight.detach().cpu().numpy() for layer in model.rq.vq_layers]
    all_code_indices = resolve_collisions(all_encoded_x, all_code_indices, codebooks)
    all_indices = []
    all_indices_str = []
    for index in all_code_indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        all_indices.append(code)
        all_indices_str.append(str(code))
    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
