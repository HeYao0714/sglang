#!/usr/bin/env bash

set -euo pipefail

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export PYTHONPATH=/home/t00937989/sglang/python:${PYTHONPATH:-}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-31000}
MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Pro-w4a8-mtp}
OUTPUT_DIR=${OUTPUT_DIR:-/home/t00937989/outputs/high_throughput_bench}

SLEEP_SECONDS=${SLEEP_SECONDS:-10}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<'EOF'
Usage:
  bench.sh DATASET_SUITE_DIR

Example:
  bench.sh /home/t00937989/datasets/gsm8k/cache90_64000

The script reads input/output lengths, prompt counts, run count, cache ratio,
and dataset filenames from DATASET_SUITE_DIR/manifest.json.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 2
fi

DATASET_SUITE_DIR=$1
MANIFEST=${DATASET_SUITE_DIR}/manifest.json

if ! command -v python3 >/dev/null 2>&1; then
  echo "Required command not found: python3" >&2
  exit 2
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

manifest_values=()
mapfile -t manifest_values < <(
  python3 - "$MANIFEST" "$DATASET_SUITE_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
suite_root = Path(sys.argv[2]).resolve()
with manifest_path.open(encoding="utf-8") as manifest_file:
    manifest = json.load(manifest_file)


def resolve_dataset(relative_name: str) -> Path:
    path = (suite_root / relative_name).resolve()
    try:
        path.relative_to(suite_root)
    except ValueError as error:
        raise SystemExit(f"Dataset path escapes suite directory: {relative_name}") from error
    if not path.is_file():
        raise SystemExit(f"Dataset file not found: {path}")
    return path


formal_files = manifest.get("formal_files")
if not isinstance(formal_files, list) or not formal_files:
    raise SystemExit("manifest.formal_files must be a non-empty list")

print(resolve_dataset(manifest["warm_file"]))
print(int(manifest["warm_prompts"]))
print(int(manifest["formal_prompts_per_run"]))
print(int(manifest["target_tokens"]))
print(int(manifest["output_tokens"]))
print(int(manifest["shared_prefix_tokens"]))
print(float(manifest["actual_content_cache_ratio"]))
for relative_name in formal_files:
    print(resolve_dataset(relative_name))
PY
)

if (( ${#manifest_values[@]} < 8 )); then
  echo "Manifest did not return the required fields: $MANIFEST" >&2
  exit 2
fi

warm_dataset=${manifest_values[0]}
warm_prompts=${manifest_values[1]}
formal_prompts=${manifest_values[2]}
target_tokens=${manifest_values[3]}
output_tokens=${manifest_values[4]}
shared_prefix_tokens=${manifest_values[5]}
actual_ratio=${manifest_values[6]}
formal_datasets=("${manifest_values[@]:7}")
manifest_runs=${#formal_datasets[@]}
NUM_RUNS=$manifest_runs
TAG=${TAG:-$(basename "$DATASET_SUITE_DIR")_${output_tokens}out}

echo "===== Cache suite ====="
echo "manifest:       $MANIFEST"
echo "target tokens:  $target_tokens"
echo "output tokens:  $output_tokens"
echo "shared prefix:  $shared_prefix_tokens"
echo "content ratio:  $actual_ratio"
echo "warm dataset:   $warm_dataset"
for ((index = 0; index < NUM_RUNS; index++)); do
  echo "formal run $((index + 1)): ${formal_datasets[$index]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, no requests were sent"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

echo "===== Warmup: ${warm_prompts} identical requests ====="
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --dataset-name sharegpt \
  --dataset-path "$warm_dataset" \
  --sharegpt-output-len "$output_tokens" \
  --num-prompts "$warm_prompts" \
  --max-concurrency "$warm_prompts" \
  --warmup-requests 0 \
  --cache-report \
  --output-file "${OUTPUT_DIR}/${TAG}_warm.json" \
  2>&1 | tee "${OUTPUT_DIR}/${TAG}_warm.log"

for ((index = 0; index < NUM_RUNS; index++)); do
  run_number=$((index + 1))
  formal_dataset=${formal_datasets[$index]}

  echo "===== Formal run ${run_number}/${NUM_RUNS}: ${formal_prompts} requests ====="
  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name sharegpt \
    --dataset-path "$formal_dataset" \
    --sharegpt-output-len "$output_tokens" \
    --num-prompts "$formal_prompts" \
    --max-concurrency "$formal_prompts" \
    --warmup-requests 0 \
    --cache-report \
    --output-file "${OUTPUT_DIR}/${TAG}_formal_run${run_number}.json" \
    2>&1 | tee "${OUTPUT_DIR}/${TAG}_formal_run${run_number}.log"

  if (( run_number < NUM_RUNS )); then
    sleep "$SLEEP_SECONDS"
  fi
done
