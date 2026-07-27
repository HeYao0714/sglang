#!/bin/bash
set -euo pipefail

unset http_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export PYTHONPATH=/home/t00937989/sglang-pd/python:${PYTHONPATH:-}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MODEL_PATH=${MODEL_PATH:-/data/weights/DeepSeek-V4-Flash-w8a8-mtp}
DATASET_NAME=${DATASET_NAME:-random}
DATASET_PATH=${DATASET_PATH:-/home/t00937989/datasets/gsm8k/test_sharegpt_style_112.json}
OUTPUT_DIR=${OUTPUT_DIR:-/home/t00937989/zkk_new/high_throughput_bench}

NUM_RUNS=${NUM_RUNS:-3}
NUM_PROMPTS=${NUM_PROMPTS:-112}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-112}
INPUT_LEN=${INPUT_LEN:-16384}
OUTPUT_LEN=${OUTPUT_LEN:-1024}
SLEEP_SECONDS=${SLEEP_SECONDS:-10}
TAG=${TAG:-16k_1k_cache0_gsm8k_conc112_112prompts_draft3_graph12468_0718}

mkdir -p "${OUTPUT_DIR}"

for i in $(seq 1 "${NUM_RUNS}"); do
  echo "===== 16k/1k 0 cache benchmark with GSM8K source run ${i}/${NUM_RUNS} ====="

  curl -fsS -X POST "http://${HOST}:${PORT}/flush_cache"

  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-path "${DATASET_PATH}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 1 \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --warmup-requests 0 \
    --cache-report \
    --output-file "${OUTPUT_DIR}/${TAG}_run${i}.json" \
    2>&1 | tee "${OUTPUT_DIR}/${TAG}_run${i}.log"

  if [ "${i}" -lt "${NUM_RUNS}" ]; then
    sleep "${SLEEP_SECONDS}"
  fi
done
