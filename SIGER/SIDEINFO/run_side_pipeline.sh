#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# ===== Stage 0: Experiment =====
SEED=42
#DATASET="Yelp"
#DATASET="Sports_and_Outdoors"
#DATASET="Toys_and_Games"
DATASET="Yelp"
MODEL="DIFF"

GPU_ID=0
RUN_NAME="02"
# ===== Stage 1: Train =====
ALPHA=0.5 #unsused
ITEM_EMB_REG_WEIGHT=0.25
ITEM_EMB_REG_TYPE="l2"
ITEM_EMB_REG_TEMP=0.1
ITEM_EMB_REG_LEARNABLE_TEMP=False
ALIGN_TEMP=0.1
LAMBDA_VALUE=100
LEARNABLE_ALIGN_TEMP=False
C_VALUE=9
VALID_METRIC="NDCG@10"
HIDDEN_DROPOUT_PROB=0.5
ATTN_DROPOUT_PROB=0.3
FUSION_TYPE="gate"

# 例子: TRAIN_EXTRA_ARGS="--epochs 300 --learning_rate 0.001"
TRAIN_EXTRA_ARGS=""
OUTPUT_DIR="./runs/${MODEL}_${DATASET}/${RUN_NAME}"
# ===== Stage 2: Extract Item Embedding =====
ITEM_EMB_PATH="${OUTPUT_DIR}/item_embeddings.npy"

# ===== Stage 3: RQ Codebook =====
RQ_N_STAGES=4
RQ_N_CLUSTERS=256
RQ_MAX_ITER=200
RQ_BALANCE_TOLERANCE=1.5
RQ_USE_FAISS=0
RQ_USE_GPU=1
RQ_DISABLE_COLLISION_RESOLUTION=0
RQ_CODES_PATH="${OUTPUT_DIR}/rq_codes.npy"
RQ_CODEBOOKS_PATH="${OUTPUT_DIR}/rq_codebooks.npz"

case "$DATASET" in
  "Beauty")
    DATASET_ID="Amazon_Beauty"
    ;;
  "Sports_and_Outdoors")
    DATASET_ID="Amazon_Sports_and_Outdoors"
    ;;
  "Toys_and_Games")
    DATASET_ID="Amazon_Toys_and_Games"
    ;;
  "Yelp")
    DATASET_ID="yelp"
    ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    exit 1
    ;;
esac

case "$MODEL" in
  "DIFF")
    CONFIG_FILE="configs/${DATASET_ID}_diff.yaml"
    ;;
  "DIFFLatePos")
    CONFIG_FILE="configs/${DATASET_ID}_diff.yaml"
    ;;
  "ASIF")
    CONFIG_FILE="configs/${DATASET_ID}_asif.yaml"
    ;;
  *)
    echo "Unsupported MODEL: $MODEL" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"

run_train() {
  set -- \
    python ./run_model.py \
    --model "$MODEL" \
    --dataset "$DATASET_ID" \
    --config_files "$CONFIG_FILE" \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --checkpoint_dir "$OUTPUT_DIR" \
    --item_emb_reg_weight "$ITEM_EMB_REG_WEIGHT" \
    --item_emb_reg_type "$ITEM_EMB_REG_TYPE" \
    --item_emb_reg_temp "$ITEM_EMB_REG_TEMP" \
    --item_emb_reg_learnable_temp "$ITEM_EMB_REG_LEARNABLE_TEMP" \
    --temp "$ALIGN_TEMP" \
    --learnable_align_temp "$LEARNABLE_ALIGN_TEMP" \
    --valid_metric "$VALID_METRIC" \
    --hidden_dropout_prob "$HIDDEN_DROPOUT_PROB" \
    --attn_dropout_prob "$ATTN_DROPOUT_PROB"

  if [ "$MODEL" = "DIFF" ] || [ "$MODEL" = "DIFFLatePos" ]; then
    set -- "$@" \
      --alpha "$ALPHA" \
      --lambda "$LAMBDA_VALUE" \
      --c "$C_VALUE" \
      --fusion_type "$FUSION_TYPE"
  else
    set -- "$@" \
      --fusion_type_early "$FUSION_TYPE" \
      --fusion_type_late "$FUSION_TYPE"
  fi

  if [ -n "$TRAIN_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2086
    set -- "$@" $TRAIN_EXTRA_ARGS
  fi

  "$@"
}

find_checkpoint() {
  for path in "$OUTPUT_DIR"/"${MODEL}-"*.pth "$OUTPUT_DIR"/*.pth; do
    if [ -f "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

run_extract() {
    set -- \
      python ./extract_embeddings.py \
      --ckpt_path "$1" \
      --output_path "$ITEM_EMB_PATH"

  "$@"
}

run_rq() {
  set -- \
    python ./rq_kmeans.py \
    --embedding_path "$ITEM_EMB_PATH" \
    --output_path "$RQ_CODES_PATH" \
    --codebooks_output "$RQ_CODEBOOKS_PATH" \
    --n_stages "$RQ_N_STAGES" \
    --n_clusters "$RQ_N_CLUSTERS" \
    --max_iter "$RQ_MAX_ITER" \
    --random_state "$SEED" \
    --balance_tolerance "$RQ_BALANCE_TOLERANCE" \
    --gpu_id "$GPU_ID" \
    --code_start_index 0

  if [ "$RQ_USE_FAISS" = "1" ]; then
    set -- "$@" --use_faiss
  fi
  if [ "$RQ_USE_GPU" = "1" ]; then
    set -- "$@" --use_gpu
  fi
  if [ "$RQ_DISABLE_COLLISION_RESOLUTION" = "1" ]; then
    set -- "$@" --disable_collision_resolution
  fi

  "$@"
}

run_train


CHECKPOINT_PATH=$(find_checkpoint || true)
if [ -z "$CHECKPOINT_PATH" ]; then
  echo "No checkpoint found in $OUTPUT_DIR" >&2
  exit 1
fi

run_extract "$CHECKPOINT_PATH"
run_rq
