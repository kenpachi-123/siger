#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

DATASETS=("Beauty" "Sports_and_Outdoors" "Toys_and_Games" "Yelp")
DEVICES=("cuda:0" "cuda:1" "cuda:0" "cuda:1")
ITEM_ID_BASE="0"
ITEM_ID_OFFSET="0"
TREE_K="${TREE_K:-256}"

prepare_one() {
  local dataset="$1"
  local device="$2"
  local gpu_id="${device#cuda:}"
  local cf_emb="/root/rivermind-data/diffandletter/DIFF/2eager/ckpt/${dataset}/cf_emb.pt"
  local sem_emb="$ROOT_DIR/ckpt/embeddings/${dataset}/${dataset}.ordered.npy"
  local dataset_dir="$ROOT_DIR/data/${dataset}"
  local quant_dir="$ROOT_DIR/quant/outputs/${dataset}"
  local prep_log="$LOG_DIR/${dataset}_prepare.log"

  mkdir -p "$dataset_dir"
  mkdir -p "$quant_dir"

  cp -f "$ROOT_DIR/../LETTER/data/${dataset}/${dataset}.inter.json" "$dataset_dir/${dataset}.inter.json"

  {
    echo "[COPY] ${dataset}.inter.json"
    echo "[QUANT] collaborative index"
    python -u "$ROOT_DIR/quant/build_eager_codes.py" \
      --tree_name collaborative \
      --collaborative_emb_path "$cf_emb" \
      --semantic_emb_path "$sem_emb" \
      --output_dir "$quant_dir" \
      --cuda_device "$gpu_id" \
      --item_id_offset "$ITEM_ID_OFFSET" \
      --k "$TREE_K"

    echo "[QUANT] semantic index"
    python -u "$ROOT_DIR/quant/build_eager_codes.py" \
      --tree_name semantic \
      --collaborative_emb_path "$cf_emb" \
      --semantic_emb_path "$sem_emb" \
      --output_dir "$quant_dir" \
      --cuda_device "$gpu_id" \
      --item_id_offset "$ITEM_ID_OFFSET" \
      --k "$TREE_K"

    cp -f "$quant_dir/eager_semantic_k${TREE_K}.index.json" "$dataset_dir/${dataset}.index.json"
    cp -f "$quant_dir/eager_collaborative_k${TREE_K}.index.json" "$dataset_dir/${dataset}.collab.index.json"
    cp -f "$quant_dir/eager_semantic_k${TREE_K}.index.json" "$dataset_dir/${dataset}.semantic.index.json"

    echo "[DONE] prepared ${dataset} with item_id_base=${ITEM_ID_BASE}"
  } > "$prep_log" 2>&1
}

pids=()
for idx in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$idx]}"
  device="${DEVICES[$idx]}"
  prepare_one "$dataset" "$device" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Prepared EAGER data for all datasets."
