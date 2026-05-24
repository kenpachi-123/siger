#!/bin/sh
set -eu
export WANDB_DISABLED=true

DATASET="${DATASET:-Beauty}"
RUN_NAME="${RUN_NAME:-eager_run}"
DEVICE="${DEVICE:-cuda:0}"

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

SEED="${SEED:-42}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-512}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
EPOCHS="${EPOCHS:-200}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BF16="${BF16:-true}"

CF_TEACHER_EMB_PATH="${CF_TEACHER_EMB_PATH:-/root/rivermind-data/diffandletter/DIFF/2eager/ckpt/${DATASET}/cf_emb.pt}"
SEM_TEACHER_EMB_PATH="${SEM_TEACHER_EMB_PATH:-${ROOT_DIR}/ckpt/embeddings/${DATASET}/${DATASET}.ordered.npy}"

ENCODER_INPUT_MODE="item_id"
REMOVE_T5_EXTRA_TOKENS="true"
ITEM_ID_BASE="0"
OPTIMIZER_IMPL="adam"
SCHEDULER_TYPE="inverse_sqrt"
WARMUP_UPDATES="2000"
WARMUP_INIT_LR="1e-7"
WEIGHT_DECAY="1e-7"

ENCODER_NUM_LAYERS="1"
DECODER_NUM_LAYERS="4"
AUX_NUM_LAYERS="1"

VALID_METRIC="${VALID_METRIC:-loss}"
VALID_METRIC_K="${VALID_METRIC_K:-20}"
VALID_SAMPLING_MODE="${VALID_SAMPLING_MODE:-fixed}"
VALID_SAMPLE_RATIO="${VALID_SAMPLE_RATIO:-1.0}"

TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
NUM_BEAMS="${NUM_BEAMS:-20}"
METRICS="${METRICS:-hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20}"
DEDUP_PREDICTIONS="${DEDUP_PREDICTIONS:-true}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-true}"
TEST_PROMPT_IDS="${TEST_PROMPT_IDS:-0}"
VALID_PROMPT_ID="${VALID_PROMPT_ID:-0}"
RESULTS_FILE="${RESULTS_FILE:-}"

BASE_MODEL="${ROOT_DIR}/T5/ckpt/TIGER"
OUTPUT_DIR="${ROOT_DIR}/ckpt/EAGER/${DATASET}/${RUN_NAME}"
DATA_PATH="${ROOT_DIR}/data"
INDEX_FILE=".index.json"
CF_INDEX_FILE="${CF_INDEX_FILE:-.collab.index.json}"
SEM_INDEX_FILE="${SEM_INDEX_FILE:-.semantic.index.json}"

if [ -z "${CF_TEACHER_EMB_PATH}" ] || [ -z "${SEM_TEACHER_EMB_PATH}" ]; then
  echo "Please set CF_TEACHER_EMB_PATH and SEM_TEACHER_EMB_PATH before running." >&2
  exit 1
fi

if [ ! -f "${CF_TEACHER_EMB_PATH}" ]; then
  echo "Missing cooperative teacher embedding: ${CF_TEACHER_EMB_PATH}" >&2
  exit 1
fi

if [ ! -f "${SEM_TEACHER_EMB_PATH}" ]; then
  echo "Missing semantic teacher embedding: ${SEM_TEACHER_EMB_PATH}" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}/${DATASET}/${DATASET}.inter.json" ]; then
  echo "Missing inter file: ${DATA_PATH}/${DATASET}/${DATASET}.inter.json" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}/${DATASET}/${DATASET}${INDEX_FILE}" ]; then
  echo "Missing index file: ${DATA_PATH}/${DATASET}/${DATASET}${INDEX_FILE}" >&2
  echo "Run ${ROOT_DIR}/prepare_eager_data.sh first." >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}/${DATASET}/${DATASET}${CF_INDEX_FILE}" ]; then
  echo "Missing collaborative index file: ${DATA_PATH}/${DATASET}/${DATASET}${CF_INDEX_FILE}" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}/${DATASET}/${DATASET}${SEM_INDEX_FILE}" ]; then
  echo "Missing semantic index file: ${DATA_PATH}/${DATASET}/${DATASET}${SEM_INDEX_FILE}" >&2
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

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}/T5"

set -- \
  python ./finetune.py \
  --base_model "${BASE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --index_file "${INDEX_FILE}" \
  --cf_index_file "${CF_INDEX_FILE}" \
  --sem_index_file "${SEM_INDEX_FILE}" \
  --seed "${SEED}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --epochs "${EPOCHS}" \
  --temperature "${TEMPERATURE}" \
  --encoder_input_mode "${ENCODER_INPUT_MODE}" \
  --item_id_base "${ITEM_ID_BASE}" \
  --cf_teacher_emb_path "${CF_TEACHER_EMB_PATH}" \
  --sem_teacher_emb_path "${SEM_TEACHER_EMB_PATH}" \
  --optimizer_impl "${OPTIMIZER_IMPL}" \
  --scheduler_type "${SCHEDULER_TYPE}" \
  --warmup_updates "${WARMUP_UPDATES}" \
  --warmup_init_lr "${WARMUP_INIT_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --encoder_num_layers "${ENCODER_NUM_LAYERS}" \
  --decoder_num_layers "${DECODER_NUM_LAYERS}" \
  --aux_num_layers "${AUX_NUM_LAYERS}" \
  --valid_metric "${VALID_METRIC}" \
  --valid_metric_k "${VALID_METRIC_K}" \
  --valid_prompt_id "${VALID_PROMPT_ID}" \
  --valid_sampling_mode "${VALID_SAMPLING_MODE}" \
  --valid_sample_ratio "${VALID_SAMPLE_RATIO}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --num_beams "${NUM_BEAMS}" \
  --test_prompt_ids "${TEST_PROMPT_IDS}" \
  --metrics "${METRICS}" \
  --results_file "${RESULTS_FILE}"

if [ "${RUN_TEST_AFTER_TRAIN}" = "false" ]; then
  set -- "$@" --no_test_after_train
fi

if [ "${DEDUP_PREDICTIONS}" = "true" ]; then
  set -- "$@" --dedup_predictions
fi

if [ "${BF16}" = "true" ]; then
  set -- "$@" --bf16
fi

if [ "${REMOVE_T5_EXTRA_TOKENS}" = "true" ]; then
  set -- "$@" --remove_t5_extra_tokens
fi

"$@"
