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
OUTPUT_DIR=${OUTPUT_DIR:-/home/t00937989/zkk_new/high_throughput_bench}

WARM_DATASET=${WARM_DATASET:-/home/t00937989/zkk_new/high_throughput_bench/autobench/128k_cache90_safe_warm16_route.jsonl}
FORMAL_DATASET=${FORMAL_DATASET:-/home/t00937989/zkk_new/high_throughput_bench/autobench/128k_cache90_safe_formal160_noroute.jsonl}

WARM_PROMPTS=${WARM_PROMPTS:-16}
WARM_CONCURRENCY=${WARM_CONCURRENCY:-16}
FORMAL_PROMPTS=${FORMAL_PROMPTS:-80}
FORMAL_CONCURRENCY=${FORMAL_CONCURRENCY:-80}
OUTPUT_LEN=${OUTPUT_LEN:-1000}
NUM_RUNS=${NUM_RUNS:-3}
SLEEP_SECONDS=${SLEEP_SECONDS:-10}
TAG=${TAG:-128k_1k_cache90_conc80_pr11_autobench_safe_0717}

mkdir -p "${OUTPUT_DIR}"

echo "===== 128k/1k 90% cache warmup: ${WARM_PROMPTS} requests ====="
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --dataset-name autobench \
  --dataset-path "${WARM_DATASET}" \
  --sharegpt-output-len "${OUTPUT_LEN}" \
  --num-prompts "${WARM_PROMPTS}" \
  --max-concurrency "${WARM_CONCURRENCY}" \
  --warmup-requests 0 \
  --return-routed-experts \
  --cache-report \
  --output-file "${OUTPUT_DIR}/${TAG}_warm.json" \
  2>&1 | tee "${OUTPUT_DIR}/${TAG}_warm.log"

for i in $(seq 1 "${NUM_RUNS}"); do
  echo "===== 128k/1k 90% cache formal run ${i}/${NUM_RUNS} ====="

  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --dataset-name autobench \
    --dataset-path "${FORMAL_DATASET}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --num-prompts "${FORMAL_PROMPTS}" \
    --max-concurrency "${FORMAL_CONCURRENCY}" \
    --warmup-requests 0 \
    --cache-report \
    --output-file "${OUTPUT_DIR}/${TAG}_formal_run${i}.json" \
    2>&1 | tee "${OUTPUT_DIR}/${TAG}_formal_run${i}.log"

  if [ "${i}" -lt "${NUM_RUNS}" ]; then
    sleep "${SLEEP_SECONDS}"
  fi
done
