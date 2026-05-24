#!/bin/sh
set -eu
export WANDB_DISABLED=true
# Data preparation
# DATASET="Sports_and_Outdoors"
# DATASET="Toys_and_Games"
# DATASET="Yelp"
DATASET="Sports_and_Outdoors"
DIFF_NAME="c=9"
RUN_NAME="19000"
CLS_WARMUP_STEPS="19000"
DEVICE="cuda:3"

RQ_CODES_NPY="/root/blockdata/diffandletter/DIFF/runs/DIFF_${DATASET}/${DIFF_NAME}/rq_codes.npy"
EMB_FILE="/root/blockdata/diffandletter/DIFF/runs/DIFF_${DATASET}/${DIFF_NAME}/item_embeddings.npy"
CODEBOOK_PATH="/root/blockdata/diffandletter/DIFF/runs/DIFF_${DATASET}/${DIFF_NAME}/rq_codebooks.npz"

# RQ_CODES_NPY="/root/blockdata/diffandletter/DIFF/runs/${RUN_NAME}/rq_codes.npy"
# EMB_FILE="/root/blockdata/diffandletter/DIFF/runs/${RUN_NAME}/item_embeddings.npy"
# CODEBOOK_PATH="/root/blockdata/diffandletter/DIFF/runs/${RUN_NAME}/rq_codebooks.npz"
OUTPUT_DIR="/root/blockdata/diffandletter_wj/LETTER/TIGER_CKPT/${DATASET}/${RUN_NAME}"



# Training
SEED="42"
PER_DEVICE_BATCH_SIZE="512"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="5e-4"
EPOCHS="200"
TEMPERATURE="1.0"
BF16="true"
REMOVE_T5_EXTRA_TOKENS="false"
# CLS contrastive
USE_CLS_CONTRA="true"
CLS_CONTRA_WEIGHT="1"
CLS_TEMPERATURE="1"
CLS_CONTRA_TYPE="infonce"         # infonce | cosine | random_neg
CLS_NUM_NEGATIVES="512"           # number of random negatives (only used when CLS_CONTRA_TYPE=random_neg)

# CLS weight scheduling
# CLS_WEIGHT_SCHEDULER: constant | linear_warmup | cosine_decay | warmup_cosine | constant_cosine
CLS_WEIGHT_SCHEDULER="constant_cosine"

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

# Init
USE_CODEBOOK_INIT="true"

# Validation during training
# VALID_METRIC: loss | ndcg | hitrate
VALID_METRIC="loss"
VALID_METRIC_K="20"
# VALID_SAMPLING_MODE: fixed | realtime
VALID_SAMPLING_MODE="fixed"
VALID_SAMPLE_RATIO="1.0"

# Test
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

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR="${ROOT_DIR}/data/${DATASET}"
INDEX_JSON="${DATA_DIR}/indices.json"
INDEX_NPY="${DATA_DIR}/indices.npy"
TRAIN_FILE="${ROOT_DIR}/ckpt/${DATASET}/train.parquet"
VALID_FILE="${ROOT_DIR}/ckpt/${DATASET}/valid.parquet"
TEST_FILE="${ROOT_DIR}/ckpt/${DATASET}/test.parquet"
BASE_MODEL="${ROOT_DIR}/LETTER-TIGER/ckpt/TIGER"
DEFAULT_OUTPUT_DIR="${ROOT_DIR}/LETTER-TIGER/ckpt/${DATASET}"
DEFAULT_EMB_FILE_CKPT="${ROOT_DIR}/ckpt/embeddings/${DATASET}/${DATASET}.ordered.npy"
DEFAULT_EMB_FILE_DATA="${DATA_DIR}/${DATASET}.ordered.npy"

mkdir -p "${DATA_DIR}"

if [ -z "${RQ_CODES_NPY}" ]; then
  echo "Please set RQ_CODES_NPY to the rq-code npy path before running this script." >&2
  exit 1
fi

if [ ! -f "${RQ_CODES_NPY}" ]; then
  echo "Missing rq-code npy: ${RQ_CODES_NPY}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VALID_FILE}" ] || [ ! -f "${TEST_FILE}" ]; then
  echo "Missing parquet files under ${ROOT_DIR}/ckpt/${DATASET}" >&2
  exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
fi

if [ -z "${RESULTS_FILE}" ]; then
  RESULTS_FILE="${OUTPUT_DIR}/test_results.json"
fi

if [ "${USE_CLS_CONTRA}" = "true" ]; then
  if [ -z "${EMB_FILE}" ]; then
    if [ -f "${DEFAULT_EMB_FILE_CKPT}" ]; then
      EMB_FILE="${DEFAULT_EMB_FILE_CKPT}"
    elif [ -f "${DEFAULT_EMB_FILE_DATA}" ]; then
      EMB_FILE="${DEFAULT_EMB_FILE_DATA}"
    else
      echo "USE_CLS_CONTRA=true but no ordered embedding file was found." >&2
      echo "Checked:" >&2
      echo "  ${DEFAULT_EMB_FILE_CKPT}" >&2
      echo "  ${DEFAULT_EMB_FILE_DATA}" >&2
      echo "Set EMB_FILE explicitly." >&2
      exit 1
    fi
  fi

  if [ ! -f "${EMB_FILE}" ]; then
    echo "Missing EMB_FILE: ${EMB_FILE}" >&2
    exit 1
  fi
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

echo "[1/4] Convert rq codes npy -> LETTER index json"
python "${ROOT_DIR}/data_process/npy2json.py" \
  --input_file "${RQ_CODES_NPY}" \
  --output_file "${INDEX_JSON}" \
  --key_base "${KEY_BASE}"

echo "[2/4] Convert LETTER index json -> integer index npy"
python "${ROOT_DIR}/RQ-VAE/trans.py" \
  --input_file "${INDEX_JSON}" \
  --output_file "${INDEX_NPY}"

echo "[3/4] Prepare LETTER-TIGER inter/index files"
ITEM_ID_BASE="${ITEM_ID_BASE}" "${ROOT_DIR}/prepare_tiger_data.sh" "${DATASET}"

echo "[4/4] Train LETTER-TIGER"
cd "${ROOT_DIR}/LETTER-TIGER"

set -- \
  python ./finetune_v1.py \
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
  --codebook_path "${CODEBOOK_PATH}"

if [ "${USE_CLS_CONTRA}" = "true" ]; then
  set -- "$@" \
    --use_cls_contra \
    --cls_contra_weight "${CLS_CONTRA_WEIGHT}" \
    --cls_temperature "${CLS_TEMPERATURE}" \
    --cls_contra_type "${CLS_CONTRA_TYPE}" \
    --cls_num_negatives "${CLS_NUM_NEGATIVES}" \
    --emb_file "${EMB_FILE}" \
    --cls_weight_scheduler "${CLS_WEIGHT_SCHEDULER}" \
    --cls_warmup_steps "${CLS_WARMUP_STEPS}"
fi

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
