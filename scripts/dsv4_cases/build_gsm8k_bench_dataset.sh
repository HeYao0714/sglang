#!/usr/bin/env bash

set -euo pipefail

SOURCE_FILE="/home/t00937989/datasets/gsm8k/test.jsonl"
TOKENIZER_PATH="/home/weights/DeepSeek-V4-Pro-w4a8-mtp"
DATASET_ROOT="/home/t00937989/datasets/gsm8k"
PAGE_SIZE="128"

target_tokens="64000"
output_tokens="1000"
cache_ratio="0.90"
shared_prefix_tokens=""
warm_prompts="2"
formal_prompts="4"
num_runs="3"
seed="20260728"
ratio_was_set=0
prefix_was_set=0

usage() {
  cat <<'EOF'
Generate a ShareGPT dataset suite with a controlled prompt-cache ratio.

Usage:
  build_gsm8k_bench_dataset.sh [OPTIONS]

Options:
  --target-tokens N           Tokens in every human prompt (default: 64000)
  --output-tokens N           Generated tokens requested by the benchmark (default: 1000)
  --cache-ratio R             Shared-prefix ratio in (0, 1) (default: 0.90)
  --shared-prefix-tokens N    Exact shared-prefix tokens; conflicts with --cache-ratio
  --warm-prompts N            Number of identical warmup prompts (default: 2)
  --formal-prompts N          Formal prompts in each run (default: 4)
  --num-runs N                Number of independent formal datasets (default: 3)
  --seed N                    Deterministic random seed (default: 20260728)
  -h, --help                  Show this help

Fixed deployment settings:
  page size:      128
  source:         /home/t00937989/datasets/gsm8k/test.jsonl
  tokenizer:      /home/weights/DeepSeek-V4-Pro-w4a8-mtp
  dataset root:   /home/t00937989/datasets/gsm8k

Example:
  build_gsm8k_bench_dataset.sh \
    --target-tokens 64000 --output-tokens 1000 --cache-ratio 0.90 \
    --warm-prompts 2 --formal-prompts 4 --num-runs 3
EOF
}

need_value() {
  if [[ "$#" -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --target-tokens)
      need_value "$@"; target_tokens="$2"; shift 2 ;;
    --output-tokens)
      need_value "$@"; output_tokens="$2"; shift 2 ;;
    --cache-ratio)
      need_value "$@"; cache_ratio="$2"; ratio_was_set=1; shift 2 ;;
    --shared-prefix-tokens)
      need_value "$@"; shared_prefix_tokens="$2"; prefix_was_set=1; shift 2 ;;
    --warm-prompts)
      need_value "$@"; warm_prompts="$2"; shift 2 ;;
    --formal-prompts)
      need_value "$@"; formal_prompts="$2"; shift 2 ;;
    --num-runs)
      need_value "$@"; num_runs="$2"; shift 2 ;;
    --seed)
      need_value "$@"; seed="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if (( ratio_was_set == 1 && prefix_was_set == 1 )); then
  echo "--cache-ratio and --shared-prefix-tokens are mutually exclusive" >&2
  exit 2
fi
if (( prefix_was_set == 1 )); then
  cache_ratio=""
fi

for item in \
  "target_tokens:$target_tokens" \
  "output_tokens:$output_tokens" \
  "warm_prompts:$warm_prompts" \
  "formal_prompts:$formal_prompts" \
  "num_runs:$num_runs" \
  "seed:$seed"; do
  name=${item%%:*}
  value=${item#*:}
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "--${name//_/-} must be a positive integer: $value" >&2
    exit 2
  fi
done

if [[ -n "$shared_prefix_tokens" ]] && ! [[ "$shared_prefix_tokens" =~ ^[1-9][0-9]*$ ]]; then
  echo "--shared-prefix-tokens must be a positive integer: $shared_prefix_tokens" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Source dataset not found: $SOURCE_FILE" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  python_bin="python"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python is required" >&2
  exit 2
fi

mkdir -p "$DATASET_ROOT"

"$python_bin" - \
  "$SOURCE_FILE" "$DATASET_ROOT" "$TOKENIZER_PATH" \
  "$target_tokens" "$output_tokens" "$cache_ratio" "$shared_prefix_tokens" "$PAGE_SIZE" \
  "$warm_prompts" "$formal_prompts" "$num_runs" "$seed" <<'PY'
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

(
    source_path,
    dataset_root_value,
    tokenizer_path,
    target_tokens_value,
    output_tokens_value,
    cache_ratio_value,
    shared_prefix_value,
    page_size_value,
    warm_prompts_value,
    formal_prompts_value,
    num_runs_value,
    seed_value,
) = sys.argv[1:]

target_tokens = int(target_tokens_value)
output_tokens = int(output_tokens_value)
page_size = int(page_size_value)
warm_prompts = int(warm_prompts_value)
formal_prompts = int(formal_prompts_value)
num_runs = int(num_runs_value)
seed = int(seed_value)
requested_ratio = float(cache_ratio_value) if cache_ratio_value else None

if requested_ratio is not None:
    if not 0 < requested_ratio < 1:
        raise ValueError(f"--cache-ratio must be in (0, 1): {requested_ratio}")
    shared_prefix_tokens = int(target_tokens * requested_ratio)
    shared_prefix_tokens -= shared_prefix_tokens % page_size
else:
    shared_prefix_tokens = int(shared_prefix_value)

if not 0 < shared_prefix_tokens < target_tokens:
    raise ValueError(
        "Shared prefix must leave both a non-empty prefix and suffix: "
        f"prefix={shared_prefix_tokens}, target={target_tokens}"
    )
if shared_prefix_tokens % page_size != 0:
    raise ValueError(
        f"Shared prefix {shared_prefix_tokens} is not aligned to page size {page_size}"
    )

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def encode(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def repeat_to_length(token_ids: list[int], length: int) -> list[int]:
    if not token_ids:
        raise ValueError("Cannot repeat an empty token sequence")
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def round_trip(token_ids: list[int]) -> tuple[str, list[int]]:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text, encode(text)


def read_source_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as source_file:
        source_text = source_file.read()
    try:
        rows = json.loads(source_text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in source_text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source dataset must contain a non-empty list of rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Every source dataset row must be a JSON object")
    return rows


def extract_question_answer(row: dict) -> tuple[str, str]:
    if isinstance(row.get("question"), str) and isinstance(row.get("answer"), str):
        return row["question"].split("\nAnswer:", 1)[0], row["answer"]
    conversations = row.get("conversations", [])
    question = next(
        (item.get("value") for item in conversations if item.get("from") == "human"),
        None,
    )
    answer = next(
        (item.get("value") for item in conversations if item.get("from") == "gpt"),
        None,
    )
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("Every source row must contain a human question and GPT answer")
    return question.split("\nAnswer:", 1)[0], answer


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(value, temporary_file, ensure_ascii=False, indent=2)
        temporary_name = temporary_file.name
    os.replace(temporary_name, path)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


rows = read_source_rows(source_path)
parsed_rows = [extract_question_answer(row) for row in rows]
rng = random.Random(seed)
rng.shuffle(parsed_rows)

required_formal = formal_prompts * num_runs
if len(parsed_rows) < required_formal + 2:
    raise ValueError(
        f"Need at least {required_formal + 2} source rows, got {len(parsed_rows)}"
    )

prefix_question, _ = parsed_rows[0]
prefix_seed = encode(prefix_question + "\n")
common_prefix_ids = repeat_to_length(prefix_seed, shared_prefix_tokens)
suffix_tokens = target_tokens - shared_prefix_tokens


def build_candidate(question: str) -> tuple[str, list[int]] | None:
    candidate_seed = encode(question + "\n")
    planned_ids = common_prefix_ids + repeat_to_length(candidate_seed, suffix_tokens)
    text, actual_ids = round_trip(planned_ids)
    if len(actual_ids) != target_tokens:
        return None
    return text, actual_ids


warm_text = None
warm_ids = None
warm_answer = None
warm_source_index = None
for source_index, (question, answer) in enumerate(parsed_rows[1:], start=1):
    candidate = build_candidate(question)
    if candidate is None:
        continue
    candidate_text, candidate_ids = candidate
    warm_text = candidate_text
    warm_ids = candidate_ids
    warm_answer = answer
    warm_source_index = source_index
    break

if warm_text is None or warm_ids is None or warm_answer is None:
    raise RuntimeError("Could not construct a token-stable warm prompt")

formal_records: list[dict] = []
formal_token_ids: list[list[int]] = []
attempted = 0
for source_index, (question, answer) in enumerate(parsed_rows[1:], start=1):
    if source_index == warm_source_index:
        continue
    attempted += 1
    candidate = build_candidate(question)
    if candidate is None:
        continue
    candidate_text, candidate_ids = candidate
    if longest_common_prefix(warm_ids, candidate_ids) != shared_prefix_tokens:
        continue
    if any(
        longest_common_prefix(previous_ids, candidate_ids) != shared_prefix_tokens
        for previous_ids in formal_token_ids
    ):
        continue
    formal_records.append(
        {
            "conversations": [
                {"from": "human", "value": candidate_text},
                {"from": "gpt", "value": answer},
            ]
        }
    )
    formal_token_ids.append(candidate_ids)
    if len(formal_records) == required_formal:
        break

if len(formal_records) != required_formal:
    raise RuntimeError(
        "Could not construct enough formal prompts with an exact shared-prefix boundary: "
        f"needed={required_formal}, built={len(formal_records)}, attempted={attempted}"
    )

warm_record = {
    "conversations": [
        {"from": "human", "value": warm_text},
        {"from": "gpt", "value": warm_answer},
    ]
}
warm_rows = [warm_record for _ in range(warm_prompts)]

if any(len(ids) != target_tokens for ids in formal_token_ids):
    raise AssertionError("Formal prompt length validation failed")
if len({prompt_hash(row["conversations"][0]["value"]) for row in formal_records}) != required_formal:
    raise AssertionError("Formal prompts are not unique")
if any(row["conversations"][0]["value"] != warm_text for row in warm_rows):
    raise AssertionError("Warm prompts are not identical")

actual_ratio = shared_prefix_tokens / target_tokens
cache_percent = int(round(actual_ratio * 100))
output_dir = (Path(dataset_root_value) / f"cache{cache_percent}_{target_tokens}").resolve()
warm_name = f"warm_{warm_prompts}_{target_tokens}.json"
atomic_json(output_dir / warm_name, warm_rows)

formal_names = []
formal_hashes = []
for run_index in range(num_runs):
    start = run_index * formal_prompts
    end = start + formal_prompts
    run_rows = formal_records[start:end]
    run_name = (
        f"formal_run{run_index + 1}_{formal_prompts}_{target_tokens}_"
        f"cache{cache_percent}.json"
    )
    atomic_json(output_dir / run_name, run_rows)
    formal_names.append(run_name)
    formal_hashes.append(
        [prompt_hash(row["conversations"][0]["value"]) for row in run_rows]
    )

manifest = {
    "format_version": 1,
    "source": str(Path(source_path).resolve()),
    "tokenizer_path": tokenizer_path,
    "seed": seed,
    "page_size": page_size,
    "target_tokens": target_tokens,
    "output_tokens": output_tokens,
    "requested_cache_ratio": requested_ratio,
    "shared_prefix_tokens": shared_prefix_tokens,
    "actual_content_cache_ratio": actual_ratio,
    "suffix_tokens": suffix_tokens,
    "warm_prompts": warm_prompts,
    "formal_prompts_per_run": formal_prompts,
    "num_runs": num_runs,
    "warm_file": warm_name,
    "formal_files": formal_names,
    "warm_prompt_sha256": prompt_hash(warm_text),
    "formal_prompt_sha256": formal_hashes,
    "validation": {
        "warm_prompts_identical": True,
        "all_prompt_tokens": target_tokens,
        "all_formal_lcp_tokens": shared_prefix_tokens,
        "formal_prompts_unique": True,
    },
}
atomic_json(output_dir / "manifest.json", manifest)

print(f"Generated cache suite in {output_dir}")
print(
    f"target={target_tokens}, shared_prefix={shared_prefix_tokens}, "
    f"suffix={suffix_tokens}, ratio={actual_ratio:.6f}"
)
print(f"warm={warm_name} ({warm_prompts} identical prompts)")
for run_index, formal_name in enumerate(formal_names, start=1):
    print(f"formal run {run_index}: {formal_name} ({formal_prompts} unique prompts)")
PY
