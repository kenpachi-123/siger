#!/bin/sh
set -eu
export WANDB_DISABLED=true

#DATASET="Sports_and_Outdoors"
#DATASET="Toys_and_Games"
#DATASET="Yelp"
DATASET="${DATASET:-Beauty}"

RUN_NAME="${RUN_NAME:-${DATASET}}"
DEVICE="${DEVICE:-cuda:1}"
BF16="${BF16:-true}"

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

PER_DEVICE_BATCH_SIZE="512"
GRADIENT_ACCUMULATION_STEPS="1"

# Training
SEED="42"

LEARNING_RATE="5e-4"
EPOCHS="200"
TEMPERATURE="1.0"

OUTPUT_DIR="${ROOT_DIR}/LETTER_CKPT/${RUN_NAME}"
REMOVE_T5_EXTRA_TOKENS="${REMOVE_T5_EXTRA_TOKENS:-false}"
USE_CODEBOOK_INIT="false"

OPTIM="adamw_torch"  #adam         optimizer_impl
WARMUP_RATIO="0.01"                #warmup_ratio
WEIGHT_DECAY="0.01"                  #weight_decay
GRAD_NORM="1.0"                      #max_grad_norm
SCHEDULER="cosine"       #constant     #lr_scheduler_type



# optim: adamw_torch -> adam
# warmup_ratio: 0.01 -> 0.0
#weight_decay: 0.01 -> 0.0
#max_grad_norm: 1.0 -> 0.0
#nlr_scheduler_type: cosine -> constant



# Validation during training
# VALID_METRIC: loss | ndcg | hitrate
VALID_METRIC="loss"
VALID_METRIC_K="20"
# VALID_SAMPLING_MODE: fixed | realtime
VALID_SAMPLING_MODE="fixed"
VALID_SAMPLE_RATIO="1.0"


DEDUP_PREDICTIONS="false"
USE_TRIE="true"
TEST_BATCH_SIZE="32"
NUM_BEAMS="20"
METRICS="hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20"

RUN_TEST_AFTER_TRAIN="true"
TEST_PROMPT_IDS="0"
RESULTS_FILE=""

KEY_BASE="0"
ITEM_ID_BASE="1"
VALID_PROMPT_ID="0"

DATA_DIR="${ROOT_DIR}/data/${DATASET}"
INDEX_JSON="${DATA_DIR}/indices.json"
INDEX_READY_JSON="${DATA_DIR}/${DATASET}.index.json"
EMB_NPY="${ROOT_DIR}/ckpt/embeddings/${DATASET}/${DATASET}.ordered.npy"
TRAIN_FILE="${ROOT_DIR}/ckpt/${DATASET}/train.parquet"
VALID_FILE="${ROOT_DIR}/ckpt/${DATASET}/valid.parquet"
TEST_FILE="${ROOT_DIR}/ckpt/${DATASET}/test.parquet"
BASE_MODEL="${ROOT_DIR}/T5/ckpt/TIGER"


mkdir -p "${DATA_DIR}"


if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VALID_FILE}" ] || [ ! -f "${TEST_FILE}" ]; then
  echo "Missing parquet files under ${ROOT_DIR}/ckpt/${DATASET}" >&2
  exit 1
fi

if [ ! -f "${EMB_NPY}" ]; then
  echo "Missing ordered embedding file: ${EMB_NPY}" >&2
  exit 1
fi

if [ ! -f "${INDEX_JSON}" ]; then
  echo "Missing indices file: ${INDEX_JSON}" >&2
  echo "Run ${ROOT_DIR}/prepare_all_indices_once.sh first." >&2
  exit 1
fi


if [ -z "${RESULTS_FILE}" ]; then
  RESULTS_FILE="${OUTPUT_DIR}/test_results.json"
fi


case "${DEVICE}" in
  cuda:*)
    CUDA_VISIBLE_DEVICES_VALUE=${DEVICE#cuda:}
    ;;
  *)
    CUDA_VISIBLE_DEVICES_VALUE="${DEVICE}"
    ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

echo "[1/3] Reuse pre-generated indices for ${DATASET}"

echo "[2/3] Prepare LETTER-TIGER inter/index files"
ITEM_ID_BASE="${ITEM_ID_BASE}" /bin/bash "${ROOT_DIR}/prepare_tiger_data.sh" "${DATASET}"

echo "[3/3] Train LETTER-TIGER"
cd "${ROOT_DIR}/T5"

set -- \
  python ./finetune.py \
  --base_model "${BASE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --data_path ../data \
  --index_file .index.json \
  --seed "${SEED}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --epochs "${EPOCHS}" \
  --temperature "${TEMPERATURE}" \
  --valid_metric "${VALID_METRIC}" \
  --valid_metric_k "${VALID_METRIC_K}" \
  --valid_prompt_id "${VALID_PROMPT_ID}" \
  --valid_sampling_mode "${VALID_SAMPLING_MODE}" \
  --valid_sample_ratio "${VALID_SAMPLE_RATIO}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --num_beams "${NUM_BEAMS}" \
  --test_prompt_ids "${TEST_PROMPT_IDS}" \
  --metrics "${METRICS}" \
  --results_file "${RESULTS_FILE}" \
  --optimizer_impl "${OPTIM}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_grad_norm "${GRAD_NORM}" \
  --lr_scheduler_type "${SCHEDULER}"

if [ "${USE_CODEBOOK_INIT}" = "true" ]; then
  set -- "$@" --use_codebook_init
fi

if [ "${RUN_TEST_AFTER_TRAIN}" = "false" ]; then
  set -- "$@" --no_test_after_train
fi

if [ "${DEDUP_PREDICTIONS}" = "true" ]; then
  set -- "$@" --dedup_predictions
fi

if [ "${USE_TRIE}" = "false" ]; then
  set -- "$@" --no_use_trie
fi

if [ "${BF16}" = "true" ]; then
  set -- "$@" --bf16
fi

if [ "${REMOVE_T5_EXTRA_TOKENS}" = "true" ]; then
  set -- "$@" --remove_t5_extra_tokens
fi

"$@"
