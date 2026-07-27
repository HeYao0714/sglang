#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SOURCE="/home/t00937989/datasets/gsm8k/test_sharegpt_style.json"
MODELSCOPE_DATASET_ID="${MODELSCOPE_DATASET_ID:-AI-ModelScope/gsm8k}"
HUGGINGFACE_DATASET_ID="${HUGGINGFACE_DATASET_ID:-openai/gsm8k}"

count="${1:-112}"
mode="${2:-sample}"
source_file="${3:-$DEFAULT_SOURCE}"
output_file="${4:-$(dirname "$source_file")/test_sharegpt_style_${count}.json}"

usage() {
  cat <<'EOF'
Usage:
  build_gsm8k_bench_dataset.sh [COUNT] [sample|repeat] [SOURCE] [OUTPUT]

Examples:
  build_gsm8k_bench_dataset.sh 112 sample
  build_gsm8k_bench_dataset.sh 3200 repeat
EOF
}

if [[ "$count" == "-h" || "$count" == "--help" ]]; then
  usage
  exit 0
fi

if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "COUNT must be a positive integer: $count" >&2
  exit 1
fi

if [[ "$mode" != "sample" && "$mode" != "repeat" ]]; then
  echo "MODE must be 'sample' or 'repeat': $mode" >&2
  usage >&2
  exit 1
fi

for command_name in jq mktemp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

if [[ "$mode" == "sample" ]] && ! command -v shuf >/dev/null 2>&1; then
  echo "Required command not found for sample mode: shuf" >&2
  exit 1
fi

if [[ ! -f "$source_file" ]]; then
  python_bin="${PYTHON_BIN:-python3}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    python_bin="python"
  fi
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Source dataset not found and Python is unavailable: $source_file" >&2
    exit 1
  fi

  source_dir=$(dirname "$source_file")
  mkdir -p "$source_dir"
  download_tmp="${source_file}.download.$$"
  trap 'rm -f "$download_tmp"' EXIT

  if ! "$python_bin" - "$download_tmp" "$MODELSCOPE_DATASET_ID" "$HUGGINGFACE_DATASET_ID" <<'PY'
import json
import sys

output_path, modelscope_id, huggingface_id = sys.argv[1:]


def write_sharegpt(dataset):
    rows = []
    for row in dataset:
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        rows.append(
            {
                "conversations": [
                    {"from": "human", "value": f"Question: {question}\nAnswer:"},
                    {"from": "gpt", "value": f" {answer}"},
                ]
            }
        )
    if not rows:
        raise RuntimeError("GSM8K dataset contains no question/answer rows")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)


errors = []
try:
    from modelscope.msdatasets import MsDataset

    for split in ("validation", "test"):
        try:
            write_sharegpt(
                MsDataset.load(
                    modelscope_id,
                    subset_name="main",
                    split=split,
                )
            )
            print(f"Downloaded GSM8K from ModelScope ({split})")
            break
        except Exception as error:
            errors.append(f"ModelScope {split}: {error}")
    else:
        raise RuntimeError("; ".join(errors))
except Exception as error:
    errors.append(f"ModelScope: {error}")
    try:
        from datasets import load_dataset

        write_sharegpt(load_dataset(huggingface_id, "main", split="test"))
        print("Downloaded GSM8K from Hugging Face")
    except Exception as fallback_error:
        errors.append(f"Hugging Face: {fallback_error}")
        raise RuntimeError("; ".join(errors)) from fallback_error
PY
  then
    echo "Unable to download GSM8K dataset" >&2
    exit 1
  fi

  mv "$download_tmp" "$source_file"
  trap - EXIT
  echo "Prepared GSM8K source dataset at $source_file"
fi

source_count=$(jq 'if type == "array" then length else error("dataset must be a JSON array") end' "$source_file")
if (( source_count == 0 )); then
  echo "Source dataset is empty: $source_file" >&2
  exit 1
fi

if [[ "$mode" == "sample" ]] && (( count > source_count )); then
  echo "Cannot sample $count unique rows from $source_count rows; use repeat mode" >&2
  exit 1
fi

output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"
tmp_file=$(mktemp "$output_dir/.build-gsm8k.XXXXXX")
trap 'rm -f "$tmp_file"' EXIT

if [[ "$mode" == "sample" ]]; then
  jq -c '.[]' "$source_file" | shuf -n "$count" | jq -s '.' >"$tmp_file"
else
  jq --argjson count "$count" \
    '[. as $source | range(0; $count) as $index | $source[$index % ($source | length)]]' \
    "$source_file" >"$tmp_file"
fi

generated_count=$(jq 'length' "$tmp_file")
if (( generated_count != count )); then
  echo "Generated row count mismatch: expected $count, got $generated_count" >&2
  exit 1
fi

mv "$tmp_file" "$output_file"
trap - EXIT
echo "Generated $generated_count rows in $output_file (mode: $mode)"
