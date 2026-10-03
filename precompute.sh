#!/usr/bin/env bash
# Precompute Qwen3-TTS Base voice profiles from reference_voices/*.wav+*.txt.
# Idempotent — skips any voice already in voices/custom_voice_manifest.json.
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="/usr/local/cuda-13.2/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda-13.2"
export TORCH_CUDA_ARCH_LIST="12.0"

exec .venv/bin/python precompute_voice.py --from-dir reference_voices --output-dir voices --mode icl "$@"
