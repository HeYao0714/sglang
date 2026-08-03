# DSV4 8k1k and 32k1k launch script update

## Scope

Update only these two existing launch scripts on the `scripts` branch:

- `scripts/dsv4_cases/8k1k0cache/run_dsv4-flash-modelslim-8k1k0cache.sh`
- `scripts/dsv4_cases/32k1k0cache/run_dsv4-flash-modelslim-32k1k0cache.sh`

The benchmark drivers and the 8k1k `before-change` snapshot remain unchanged.

## Source of truth

Use the complete 8k1k and 32k1k script bodies supplied by the user. Preserve their case-specific settings, including bootstrap address, compressor-prefill mode, graph batch sizes, maximum running requests, and prefill request limits.

Normalize only two clear transcription issues:

- Change the 32k1k value `SGLANG_OPT_DEEPGEMM_HC_PRENORM=Fals` to `False`.
- Render the explanatory comment as `MQALayer.__init__` instead of the Markdown-transformed `MQALayer.**init**`.

Do not introduce a shared template or unrelated refactoring.

## Verification

- Run `bash -n` on both changed scripts.
- Assert the important 8k1k and 32k1k case-specific values from the approved inputs.
- Inspect the Git diff and confirm that only the two launch scripts changed in the implementation commit.

The local environment does not provide the target Ascend NPU runtime, model weights, or cluster network. Verification therefore covers shell syntax and configuration fidelity, not a live SGLang launch.
