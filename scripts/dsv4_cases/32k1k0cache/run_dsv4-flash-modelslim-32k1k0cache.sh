#!/bin/bash
unset http_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
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
export USE_NPU_MOE_GATING_TOP_K=${USE_NPU_MOE_GATING_TOP_K:-1}
export SGLANG_NPU_USE_MULTI_STREAM=${SGLANG_NPU_USE_MULTI_STREAM:-1}
# export SGLANG_SET_CPU_AFFINITY=1
# export HCCL_SOCKET_IFNAME=lo
# export GLOO_SOCKET_IFNAME=lo
# export HCCL_OP_EXPANSION_MODE=AIV
# export HCCL_DETERMINISTIC=true
# export TASK_QUEUE_ENABLE=1

# deepep
export HCCL_BUFFSIZE=1000
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64

# zbal
if [ "${ENABLE_ZBAL:-1}" = "1" ]; then
    export HCCL_BUFFSIZE=8
    unset PYTORCH_NPU_ALLOC_CONF
    export SGLANG_ZBAL_LOCAL_MEM_SIZE=61000
    export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
    # zbccl if use mix alloc
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
    export SGLANG_ZBAL_BOOTSTRAP_URL="${SGLANG_ZBAL_BOOTSTRAP_URL:-tcp://80.5.17.40:24699}"
    export ZBAL_ENABLE_GRAPH=1
fi

# dsv4
export IS_DEEPSEEK_V4=1
export SGLANG_DEBUG_LAYER_NORM=1
export SGLANG_DEBUG_FWD_INPUT=1
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=0

# skip gpu branch
# W8A8 (modelslim) weights don't create blockwise-FP8 weight_scale_inv,
# so the FP8 wo_a GEMM opt must be off or MQALayer.__init__ asserts.
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

# mtp
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export SGLANG_NPU_PROFILING=0
export SGLANG_DEBUG_MTP_VERIFY=${SGLANG_DEBUG_MTP_VERIFY:-0}
export SGLANG_DEBUG_MTP_VERIFY_LIMIT=${SGLANG_DEBUG_MTP_VERIFY_LIMIT:-8}
export SGLANG_DEBUG_MTP_VERIFY_ROWS=${SGLANG_DEBUG_MTP_VERIFY_ROWS:-4}
# DSV4 NPU multi-token draft-extend graph is gated in code; keep this as a
# manual override instead of disabling all draft-extend graph paths by default.
export SGLANG_DISABLE_DRAFT_EXTEND_GRAPH=${SGLANG_DISABLE_DRAFT_EXTEND_GRAPH:-1}

# path
MODEL_PATH=${MODEL_PATH:-/home/weights/deepseekv4-flash-w8a8-mtp}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-131072}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-"1 2 4 8"}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
ENABLE_OVERLAP_SCHEDULE=${ENABLE_OVERLAP_SCHEDULE:-1}
SPECULATIVE_ALGORITHM=${SPECULATIVE_ALGORITHM:-EAGLE}
SPECULATIVE_NUM_STEPS=${SPECULATIVE_NUM_STEPS:-2}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-3}
EXTRA_ARGS=${EXTRA_ARGS:-"--ep-size 16 --disable-radix-cache"}

if [ "${ENABLE_OVERLAP_SCHEDULE:-0}" = "1" ]; then
    OVERLAP_ARGS=""
else
    OVERLAP_ARGS="--disable-overlap-schedule"
fi

if [ "${DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
    CUDA_GRAPH_ARGS="--disable-cuda-graph"
else
    CUDA_GRAPH_ARGS="--cuda-graph-bs ${CUDA_GRAPH_BS}"
fi

if [ "${ENABLE_MTP:-1}" = "1" ]; then
    MTP_ARGS="--speculative-algorithm ${SPECULATIVE_ALGORITHM} \
        --speculative-num-steps ${SPECULATIVE_NUM_STEPS} \
        --speculative-eagle-topk ${SPECULATIVE_EAGLE_TOPK:-1} \
        --speculative-num-draft-tokens ${SPECULATIVE_NUM_DRAFT_TOKENS}"
else
    MTP_ARGS=""
fi

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --prefill-max-requests 32 \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
    --max-running-requests ${MAX_RUNNING_REQUESTS} \
    ${OVERLAP_ARGS} \
    --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim --enable-dp-lm-head \
    --kv-cache-dtype auto \
    --skip-server-warmup \
    ${CUDA_GRAPH_ARGS} ${MTP_ARGS} ${EXTRA_ARGS:-}
