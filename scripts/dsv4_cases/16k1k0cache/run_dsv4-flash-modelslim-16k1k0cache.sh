#!/bin/bash

unset http_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1

# Keep the single-stream behavior of the historical baseline case.
export USE_NPU_MOE_GATING_TOP_K=0
export SGLANG_NPU_USE_MULTI_STREAM=0

# DeepEP
export HCCL_BUFFSIZE=1000
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

# ZBAL
export HCCL_BUFFSIZE=8
unset PYTORCH_NPU_ALLOC_CONF
export SGLANG_ZBAL_LOCAL_MEM_SIZE=61000
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
export SGLANG_ZBAL_BOOTSTRAP_URL=tcp://192.168.0.192:24699
export ZBAL_ENABLE_GRAPH=1

# DeepSeek V4 NPU
export IS_DEEPSEEK_V4=1
export SGLANG_DEBUG_LAYER_NORM=1
export SGLANG_DEBUG_FWD_INPUT=1
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=0

# Skip GPU-only optimization paths.
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False

# MTP debug and draft-extend graph controls.
export SGLANG_NPU_PROFILING=0
export SGLANG_DEBUG_MTP_VERIFY=0
export SGLANG_DEBUG_MTP_VERIFY_LIMIT=8
export SGLANG_DEBUG_MTP_VERIFY_ROWS=4
export SGLANG_DISABLE_DRAFT_EXTEND_GRAPH=0

export PYTHONPATH=/home/t00937989/sglang-pd/python:${PYTHONPATH:-}

python3 -m sglang.launch_server \
    --model-path /data/weights/DeepSeek-V4-Flash-w8a8-mtp \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --prefill-max-requests 2 \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 65536 \
    --max-running-requests 128 \
    --dp-size 16 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --quantization modelslim \
    --enable-dp-lm-head \
    --kv-cache-dtype auto \
    --skip-server-warmup \
    --cuda-graph-bs 1 2 4 6 8 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3 \
    --ep-size 16
