#!/usr/bin/env bash
# Wrapper that pins the demo to GPU 1 (RTX 5090 on this host) and the
# CUDA 13.2 toolkit (required for FlashInfer sm_120 JIT compile).
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# CUDA 13.2 toolkit on PATH — FlashInfer's JIT compiler probes nvcc to
# decide whether sm_120 kernels can be emitted. anaconda's nvcc 12.8 is
# rejected because Blackwell needs CUDA >= 12.9.
export PATH="/usr/local/cuda-13.2/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda-13.2"

# Pin arch list to Blackwell to short-circuit any multi-arch compile.
export TORCH_CUDA_ARCH_LIST="12.0"

exec .venv/bin/python demo.py "$@"
