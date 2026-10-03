#!/usr/bin/env bash
# Interactive Base-model voice-cloning REPL. Same env as run.sh.
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export PATH="/usr/local/cuda-13.2/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda-13.2"
export TORCH_CUDA_ARCH_LIST="12.0"

exec .venv/bin/python clone_repl.py "$@"
