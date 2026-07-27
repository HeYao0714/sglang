#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SOURCE="/home/t00937989/datasets/gsm8k/test.jsonl"

sample_count="${1:-112}"
target_tokens="${2:-8192}"
source_file="${3:-$DEFAULT_SOURCE}"
output_file="${4:-$(dirname "$source_file")/test_sharegpt_style_${sample_count}_${target_tokens}.json}"

usage() {
  cat <<'EOF'
Usage:
  TOKENIZER_PATH=/path/to/tokenizer build_gsm8k_bench_dataset.sh \
    SAMPLE_COUNT TARGET_TOKENS [SOURCE] [OUTPUT]

Examples:
  TOKENIZER_PATH=/models/deepseek \
    build_gsm8k_bench_dataset.sh 112 8192
  TOKENIZER_PATH=/models/deepseek \
    build_gsm8k_bench_dataset.sh 112 16384 \
    /data/gsm8k/test_sharegpt_style.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! [[ "$sample_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAMPLE_COUNT must be a positive integer: $sample_count" >&2
  exit 1
fi

if ! [[ "$target_tokens" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_TOKENS must be a positive integer: $target_tokens" >&2
  exit 1
fi

for command_name in jq mktemp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  echo "TOKENIZER_PATH must point to a local tokenizer" >&2
  usage >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  python_bin="python"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python is required for tokenizer and dataset conversion" >&2
  exit 1
fi

if [[ ! -f "$source_file" ]]; then
  echo "Source dataset not found: $source_file" >&2
  exit 1
fi

output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"
tmp_file=$(mktemp "$output_dir/.build-gsm8k.XXXXXX")
trap 'rm -f "$tmp_file"' EXIT

"$python_bin" - "$source_file" "$tmp_file" "$sample_count" "$target_tokens" "$TOKENIZER_PATH" <<'PY'
import json
import random
import sys

from transformers import AutoTokenizer

source_path, output_path, sample_count, target_tokens, tokenizer_path = sys.argv[1:]
sample_count = int(sample_count)
target_tokens = int(target_tokens)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
with open(source_path, encoding="utf-8") as source_file:
    if source_path.endswith(".jsonl"):
        source_rows = [json.loads(line) for line in source_file if line.strip()]
    else:
        source_rows = json.load(source_file)

if not source_rows:
    raise RuntimeError("Source dataset is empty")
if sample_count <= len(source_rows):
    rows = random.sample(source_rows, sample_count)
else:
    rows = random.choices(source_rows, k=sample_count)
output_rows = []
for row in rows:
    if "question" in row and "answer" in row:
        question = row["question"]
        answer = row["answer"]
    else:
        conversations = row.get("conversations", [])
        question = next(
            (item["value"] for item in conversations if item.get("from") == "human"),
            None,
        )
        answer = next(
            (item["value"] for item in conversations if item.get("from") == "gpt"),
            None,
        )
    if not isinstance(question, str) or not isinstance(answer, str):
        raise RuntimeError("Every row must contain human and gpt conversations")

    question = question.split("\nAnswer:", 1)[0]
    marker = "\nAnswer:"
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    if len(marker_tokens) >= target_tokens:
        raise RuntimeError("TARGET_TOKENS is too small for the Answer marker")

    available = target_tokens - len(marker_tokens)
    repeated = (question_tokens * ((available // len(question_tokens)) + 1))[:available]
    prompt = tokenizer.decode(repeated + marker_tokens, skip_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) != target_tokens:
        raise RuntimeError(
            f"Tokenizer could not produce exact length: expected {target_tokens}, "
            f"got {len(prompt_tokens)}"
        )

    output_rows.append(
        {"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]}
    )

with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(output_rows, output_file, ensure_ascii=False, indent=2)
PY

generated_count=$(jq 'length' "$tmp_file")
if (( generated_count != sample_count )); then
  echo "Generated row count mismatch: expected $sample_count, got $generated_count" >&2
  exit 1
fi

mv "$tmp_file" "$output_file"
trap - EXIT
echo "Generated $generated_count rows with $target_tokens input tokens in $output_file"
