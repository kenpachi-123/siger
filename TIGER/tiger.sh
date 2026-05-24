#!/bin/sh
set -eu
export WANDB_DISABLED=true

# DATASET="Sports_and_Outdoors"
# DATASET="Toys_and_Games"
# DATASET="Yelp"
#DATASET="Beauty"

DATASET="${DATASET:-Sports_and_Outdoors}"
RUN_NAME="${RUN_NAME:-${DATASET}}"
DEVICE="${DEVICE:-cuda:3}"
DEDUP_PREDICTIONS="false"
USE_TRIE="true"
BF16="${BF16:-true}"
REMOVE_T5_EXTRA_TOKENS="${REMOVE_T5_EXTRA_TOKENS:-false}"

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DECOR_DIR="${ROOT_DIR}/../DECOR"

SEED="42"
PER_DEVICE_BATCH_SIZE="512"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="5e-4"
EPOCHS="200"
TEMPERATURE="1.0"

OPTIM="adamw_torch"
WARMUP_RATIO="0.01"
WEIGHT_DECAY="0.01"
GRAD_NORM="1.0"
SCHEDULER="cosine"

VALID_METRIC="loss"
VALID_METRIC_K="20"
VALID_SAMPLING_MODE="fixed"
VALID_SAMPLE_RATIO="1.0"

TEST_BATCH_SIZE="32"
NUM_BEAMS="20"
METRICS="hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20"
RUN_TEST_AFTER_TRAIN="true"
TEST_PROMPT_IDS="0"
RESULTS_FILE=""
VALID_PROMPT_ID="0"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/ckpt/tiger/${RUN_NAME}}"
DATA_DIR="${ROOT_DIR}/data/${DATASET}"
DECOR_INDEX_JSON="${DECOR_INDEX_JSON:-${DECOR_DIR}/data/${DATASET}/${DATASET}.index.json}"
LOCAL_INDEX_JSON="${LOCAL_INDEX_JSON:-${DATA_DIR}/${DATASET}.index.json}"
TRAIN_FILE="${ROOT_DIR}/ckpt/${DATASET}/train.parquet"
VALID_FILE="${ROOT_DIR}/ckpt/${DATASET}/valid.parquet"
TEST_FILE="${ROOT_DIR}/ckpt/${DATASET}/test.parquet"
BASE_MODEL="${ROOT_DIR}/T5/ckpt/TIGER"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
AUTO_RESUME="${AUTO_RESUME:-false}"

if [ -z "${RESUME_FROM_CHECKPOINT}" ] && [ "${AUTO_RESUME}" = "true" ]; then
  LATEST_CHECKPOINT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
  if [ -n "${LATEST_CHECKPOINT}" ]; then
    RESUME_FROM_CHECKPOINT="${LATEST_CHECKPOINT}"
    echo "Auto-selected resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
  fi
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ] && [ ! -d "${RESUME_FROM_CHECKPOINT}" ]; then
  echo "Resume checkpoint directory not found: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi






mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VALID_FILE}" ] || [ ! -f "${TEST_FILE}" ]; then
  echo "Missing parquet files under ${ROOT_DIR}/ckpt/${DATASET}" >&2
  exit 1
fi

if [ ! -f "${DECOR_INDEX_JSON}" ]; then
  echo "Missing DECOR index file: ${DECOR_INDEX_JSON}" >&2
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

echo "[1/2] Copy DECOR index.json into TIGER data dir"
cp "${DECOR_INDEX_JSON}" "${LOCAL_INDEX_JSON}"

if [ ! -f "${LOCAL_INDEX_JSON}" ]; then
  echo "Failed to copy index file to ${LOCAL_INDEX_JSON}" >&2
  exit 1
fi

echo "[2/2] Train TIGER T5 with DECOR-generated index"
cd "${ROOT_DIR}/T5"
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
  --lr_scheduler_type "${SCHEDULER}" \
  $( [ -n "${RESUME_FROM_CHECKPOINT}" ] && printf '%s %s ' --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" )\
  $( [ "${RUN_TEST_AFTER_TRAIN}" = "false" ] && printf '%s ' --no_test_after_train )\
  $( [ "${DEDUP_PREDICTIONS}" = "true" ] && printf '%s ' --dedup_predictions )\
  $( [ "${USE_TRIE}" = "false" ] && printf '%s ' --no_use_trie )\
  $( [ "${BF16}" = "true" ] && printf '%s ' --bf16 )\
  $( [ "${REMOVE_T5_EXTRA_TOKENS}" = "true" ] && printf '%s ' --remove_t5_extra_tokens )
