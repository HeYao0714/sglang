#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SOURCE="/home/t00937989/datasets/gsm8k/test_sharegpt_style.json"

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
  echo "Source dataset not found: $source_file" >&2
  exit 1
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
