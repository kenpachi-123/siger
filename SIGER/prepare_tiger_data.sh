
#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ITEM_ID_BASE="${ITEM_ID_BASE:-1}"

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("Beauty" "Sports_and_Outdoors" "Toys_and_Games" "Yelp")
fi

echo "Preparing LETTER-TIGER inputs with ITEM_ID_BASE=${ITEM_ID_BASE}"

for DATASET in "${DATASETS[@]}"; do
  TRAIN_FILE="./ckpt/${DATASET}/train.parquet"
  VALID_FILE="./ckpt/${DATASET}/valid.parquet"
  TEST_FILE="./ckpt/${DATASET}/test.parquet"
  DATA_DIR="./data/${DATASET}"
  INDEX_SRC="${DATA_DIR}/indices.json"
  INDEX_DST="${DATA_DIR}/${DATASET}.index.json"

  if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VALID_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "Missing parquet files for ${DATASET} under ./ckpt/${DATASET}" >&2
    exit 1
  fi

  if [ ! -f "$INDEX_SRC" ] && [ ! -f "$INDEX_DST" ]; then
    echo "Missing index json for ${DATASET}. Expected ${INDEX_SRC} or ${INDEX_DST}" >&2
    exit 1
  fi

  python ./data_process/parquet_to_letter_inter.py \
    --dataset "${DATASET}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --test_file "${TEST_FILE}" \
    --item_id_base "${ITEM_ID_BASE}" \
    --validate_prefix \
    --output_root ./data

  if [ -f "$INDEX_SRC" ]; then
    cp -f "$INDEX_SRC" "$INDEX_DST"
  fi

  echo "Prepared ${DATASET}: ./data/${DATASET}/${DATASET}.inter.json"
  echo "Prepared ${DATASET}: ./data/${DATASET}/${DATASET}.index.json"
done
