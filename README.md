# Qwen3-TTS Streaming Demo (RTX 5090 / vLLM-Omni)

Reproduce Qwen team's **"97 ms time-to-first-audio"** claim for Qwen3-TTS on a consumer Blackwell GPU, and extend it with precomputed-x-vector voice cloning.

Measured on RTX 5090:

| Mode | Model | TTFA (p50, warm) | Reference |
|---|---|---|---|
| **CustomVoice** (predefined speakers) | 0.6B-CustomVoice | **47 ms** | Qwen claims 97 ms, GPU unspecified |
| **Base + precomputed voice** (clone) | 0.6B-Base | **59 ms** | — |
| Transformers-native (sibling bench) | 1.7B-Base | 2300 ms | 25× slower, no pipelining |

**Want to hear the samples without installing anything?** Open [`showcase/`](showcase/) — three subdirectories with ready-to-play 24 kHz WAVs:

- [`showcase/customvoice_native/`](showcase/customvoice_native/) — one sample per CustomVoice speaker in their native language
- [`showcase/customvoice_english/`](showcase/customvoice_english/) — the same English prompt through all 9 CustomVoice speakers (mix of M & F, native English + cross-language)
- [`showcase/clone/`](showcase/clone/) — three utterances per voice from the voice-clone bench, five voices (Qwen's reference + four OpenAI-sourced clones)

The gap between the vLLM-Omni path (sub-100 ms) and the transformers-native path (seconds) is the inference architecture: vLLM-Omni runs a two-stage Talker→Code2Wav pipeline through a shared-memory connector, so the first PCM chunk ships while the Talker is still generating later tokens. Transformers waits for the whole utterance.

If you came for the audio samples, jump to [Listening to the output](#listening-to-the-output).

**→ Companion write-up (coming soon):** [storminthecastle.com/posts/qwen_tts_latency/](https://www.storminthecastle.com/posts/qwen_tts_latency/)
## Prerequisites

| Component | Required | Notes |
|---|---|---|
| OS | Linux (tested on Ubuntu 24.04) | Windows unsupported upstream |
| GPU | NVIDIA Blackwell (sm_120, e.g. RTX 5090 / RTX PRO 6000) | sm_80+ works for vLLM but the demo pins arch=12.0 |
| GPU memory | ≥10 GB free | 0.6B weights ~2 GB + CUDA graphs ~5 GB + KV cache |
| Driver | ≥580 (CUDA 13-capable) | nvidia-smi should report `CUDA Version: 13.x` |
| CUDA toolkit | 13.0+ installed at `/usr/local/cuda-13.x` | needed for FlashInfer JIT compile |
| `uv` | 0.4+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Disk | ~15 GB | torch cu130 + vllm + vllm-omni + model weights |

Non-Blackwell GPUs (Hopper, Ampere) work — change `TORCH_CUDA_ARCH_LIST` to match your arch, keep the rest.

## Quick start

```bash
# 1. Clone + env (one-time, ~5 min)
git clone <this-repo> qwen3-tts-stream-demo && cd qwen3-tts-stream-demo

uv venv --python 3.12.11 --seed --managed-python
uv pip install vllm==0.30.0 --torch-backend=auto --python .venv/bin/python
VLLM_OMNI_TARGET_DEVICE=cuda uv pip install vllm-omni==0.30.0 sounddevice openai --python .venv/bin/python

# 2. Run the TTFA benchmark on CustomVoice (3 warmup + 3 timed runs; ~8 min first time)
./run.sh --n-runs 3
#   → output_audio/warmup.wav, run_00.wav, run_01.wav, run_02.wav
#   → summary table with TTFA min / median / max

# 3. Interactive CustomVoice REPL — type text, hear it stream through the speakers
./repl.sh
```

The one-time cost on first run (~5-8 min) is:
- torch 2.13+cu130 wheel download (~2 GB)
- model download (0.6B weights, ~1 GB)
- FlashInfer kernel autotune (one bucket per batch size)
- CUDA-graph capture across ~128 shapes

Everything is cached under `~/.cache/vllm/` and `$HF_HOME` after the first run.

## The three paths

### 1. Streaming TTFA demo — `demo.py` / `run.sh`

Headline benchmark: warmup pass + N measurement runs, prints per-chunk latency plus a summary. The TTFA you see here is comparable to Qwen's upstream claim.

```bash
./run.sh --text "any English text" --speaker Ryan --n-runs 5
```

Flags: `--model`, `--text`, `--speaker`, `--language`, `--output-dir`, `--n-runs`, `--debug`.

### 2. Multi-voice sample generator — `multi_voice.py`, `english_multi_voice.py`

Loops through all 9 CustomVoice speakers:

| Speaker | Gender | Native language | Description |
|---|---|---|---|
| Vivian | F | Chinese | Bright young female |
| Serena | F | Chinese | Warm, gentle young female |
| Uncle_Fu | M | Chinese | Seasoned male, mellow timbre |
| Dylan | M | Chinese (Beijing) | Youthful Beijing male |
| Eric | M | Chinese (Sichuan) | Lively Chengdu male |
| **Ryan** | **M** | **English** | Dynamic male voice with rhythm |
| **Aiden** | **M** | **English** | Sunny American male |
| Ono_Anna | F | Japanese | Playful Japanese female |
| Sohee | F | Korean | Warm Korean female |

```bash
# Native-language sample per speaker (what each voice was tuned for)
.venv/bin/python multi_voice.py           # → samples/<Speaker>_<Lang>.wav

# Same English text through every speaker (cross-language accents)
.venv/bin/python english_multi_voice.py   # → samples_english/<Speaker>_English.wav
```

**Native-English speakers: Ryan, Aiden — both male.** For English female voices, see the voice-cloning path.

### 3. Voice cloning with precomputed x-vectors — `generate_reference_voices.py` → `precompute_voice.py` → `clone_bench.py` / `clone_repl.py`

The Base variant of Qwen3-TTS does in-context voice cloning from a reference audio clip + transcript. The naive path (ref audio in every request) adds ~50-80 ms of codec encode to each utterance. This project implements the production pattern: **precompute** each voice's speaker x-vector once, drop the safetensors into a directory, and point the engine at it via `custom_voice_dir` in the deploy config. The engine loads everything into its `SpeakerEmbeddingCache` at startup, so request-time cost is a dict lookup. TTFA matches CustomVoice to within ~10 ms.

```bash
# 3a. Generate 4 reference clips (2 M, 2 F) via OpenAI TTS. Needs your API key.
#     Reads OPENAI_API_KEY from env — never written to disk or logged.
export OPENAI_API_KEY='sk-...'
.venv/bin/python generate_reference_voices.py
# → reference_voices/oai_{echo,onyx,nova,shimmer}.wav + .txt

# 3b. Precompute speaker x-vector + ref_code for each clip (idempotent).
./precompute.sh
# → voices/oai_*.safetensors + voices/custom_voice_manifest.json

# 3c. Benchmark TTFA across every precomputed voice.
./clone_bench.sh --n-runs 3
# → samples_clone/<voice>_<n>.wav + summary table

# 3d. Interactive cloning REPL.
./clone_repl.sh
```

Example `clone_repl.sh` session:
```
[oai_echo/English]> Hello, this is a cloned voice.
  TTFA=59.3ms  total=612.4ms  audio=2.80s  (voice=oai_echo)
[oai_echo/English]> /voice oai_nova
  switched to oai_nova (mode=icl)
[oai_nova/English]> Different speaker now.
  TTFA=57.8ms  total=413.9ms  audio=2.08s  (voice=oai_nova)
[oai_nova/English]> /quit
```

Adding a new voice at runtime isn't supported (the offline AsyncOmni doesn't expose IPC into the engine's cache). Workflow is `/quit` → drop WAV+TXT in `reference_voices/` → `./precompute.sh` → `./clone_repl.sh`.

## Listening to the output

After following the quick start you'll have your own outputs under the directories the scripts write to by default:

| Script output dir | What's inside | How to make | Committed? |
|---|---|---|---|
| `output_audio/` | Benchmark runs from `demo.py` | `./run.sh` | gitignored |
| `samples/` | Native-language sample per CustomVoice speaker | `multi_voice.py` | gitignored |
| `samples_english/` | Same English text per CustomVoice speaker | `english_multi_voice.py` | gitignored |
| `samples_clone/` | Per-voice samples from precomputed clones | `./clone_bench.sh` | gitignored |

A frozen set of reference outputs — generated during the write-up of this project — lives under `showcase/`:

```
showcase/
  customvoice_native/     — one WAV per CustomVoice speaker in their native language
  customvoice_english/    — the same English prompt through all 9 speakers (M+F mix,
                            including the cross-language female voices for comparison)
  clone/                  — 3 WAVs per voice from the Base-model voice-clone bench:
                            qwen_sample_m (Qwen's own ref) + 4 OpenAI-sourced voices
                            (oai_echo, oai_onyx, oai_nova, oai_shimmer)
```

Re-running the scripts regenerates fresh outputs into the gitignored script dirs without touching `showcase/`. All files are 24 kHz 16-bit PCM WAV; play with `aplay`, `vlc`, Finder preview, etc.

## What's the catch? Three non-obvious things

If you reproduce this from scratch, you will probably hit all three:

1. **`full_utterance_decode: [False]`** in the per-request `additional_information`. vLLM-Omni's example `end2end.py` defaults this to `True` because it optimises throughput — but `True` **defers Code2Wav emission until the Talker completes the full sequence**, which disables streaming entirely. For TTFA you need `False`. Gating is at `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py:222–244`.
2. **CUDA 13 toolkit on `PATH` ahead of CUDA 12.x.** FlashInfer's JIT kernel generator rejects sm_120 if nvcc on `PATH` is < 12.9 (Blackwell requires CUDA ≥ 12.9). Symptom: `RuntimeError: FlashInfer requires GPUs with sm75 or higher`. The wrappers (`run.sh` etc.) set `PATH=/usr/local/cuda-13.2/bin:$PATH` for exactly this reason.
3. **uv-managed Python 3.12 (not system Python).** Debian/Ubuntu's `/usr/include/python3.12/pyconfig.h` chains to `<x86_64-linux-gnu/python3.12/pyconfig.h>` and Triton's JIT compiler doesn't put the multi-arch include dir on the command line, so JIT fails with `fatal error: pyconfig.h: No such file`. uv's prebuilt CPython has a self-contained header.

The `MultimodalPayload` returned by `AsyncOmni.generate` is a `Mapping` subclass, not a `dict` — `isinstance(mm, dict)` returns False, which will silently drop chunks if you check for a dict. Use `mm.get("audio")` or `hasattr(mm, "get")`.

## Deploy configs

| File | Where it lives | What it does |
|---|---|---|
| `qwen3_tts_mrv2.yaml` | shipped in `vllm_omni/deploy/` | Model Runner V2 + `talker_first_audio: true` + two-stage pipeline. Used by `demo.py`, `multi_voice.py`, `repl.py`. |
| `qwen3_tts_base_voices.yaml` | this repo | Inherits mrv2 + adds `custom_voice_dir: voices` on the Talker. Used by `clone_bench.py`, `clone_repl.py`. |
| `qwen3_tts_fused_single_gpu.yaml` | **not in 0.30.0 wheel** | Single-stage fused Talker w/ in-Talker streaming codec decoder. Lands next release; expect further TTFA reduction. |

The `base_config:` reference in `qwen3_tts_base_voices.yaml` is an absolute path into the venv's `vllm_omni/deploy/`. If you move the venv, re-run `realpath .venv/.../vllm_omni/deploy/qwen3_tts_mrv2.yaml` and update the YAML.

## Repo layout

```
demo.py                        streaming TTFA harness (CustomVoice)
multi_voice.py                 one sample per speaker, native language
english_multi_voice.py         same English text through all 9 speakers
repl.py                        interactive REPL (CustomVoice)
run.sh / repl.sh               wrappers that pin GPU, CUDA toolkit, arch

# Voice-cloning (Base-model) path
generate_reference_voices.py   OpenAI TTS → reference_voices/*.wav + .txt
precompute_voice.py            Base → voices/*.safetensors + manifest
clone_bench.py                 per-voice TTFA benchmark
clone_repl.py                  interactive cloning REPL
precompute.sh / clone_bench.sh / clone_repl.sh   wrappers
qwen3_tts_base_voices.yaml     Base deploy config (mrv2 + custom_voice_dir)

reference_voices/              ref clips (safe to commit — OpenAI output)
voices/                        precomputed voice profiles (safetensors + manifest)
showcase/                      committed reference outputs — safe to listen to
    customvoice_native/        one WAV per speaker in their native language
    customvoice_english/       same English text through all 9 speakers
    clone/                     3 WAVs per voice from the voice-clone bench
samples/, samples_english/,
samples_clone/, output_audio/  default script output locations (gitignored —
                               re-running the scripts here won't touch showcase/)

pyproject.toml                 dep pins; `.venv/` is gitignored
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TTFA=nan chunks=0`, 1 yield total | `full_utterance_decode=True` or Mapping/dict typecheck bug | Set `full_utterance_decode=False`; drop `isinstance(mm, dict)` guards |
| `RuntimeError: FlashInfer requires GPUs with sm75 or higher` | `nvcc` on PATH is < 12.9 | `export PATH=/usr/local/cuda-13.2/bin:$PATH` (the wrappers do this) |
| `fatal error: x86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory` | Triton JIT building against system Python 3.12 | Create venv with uv-managed Python: `uv venv --python 3.12.11 --managed-python` |
| Hangs on `AsyncOmniEngine initialized in …` for ~5 min | CUDA graph capture + FlashInfer autotune on cold cache | Expected on first start; subsequent runs reuse `~/.cache/vllm/` |
| `FileNotFoundError: .../qwen3_tts_mrv2.yaml` | `base_config:` resolves relative to the YAML file, not the venv | Edit `qwen3_tts_base_voices.yaml` and update the absolute path |
| `ERROR: OPENAI_API_KEY not set` | OpenAI step without the env var | `export OPENAI_API_KEY='sk-...'` before running `generate_reference_voices.py` |
| `custom_voice_dir ... manifest not found` | voices/ hasn't been precomputed yet | Run `./precompute.sh` first |
| Blackwell is detected but TTFA is seconds, not ms | `qwen3_tts.yaml` default profile vs mrv2 | The demo's `DEPLOY_CONFIG_NAME` in `demo.py` must be `qwen3_tts_mrv2.yaml` |
| GPU 0 is busy / wrong GPU picked | Env defaults to GPU 1 for the author's host | Override `CUDA_VISIBLE_DEVICES` before running the wrappers |

## Credits & attribution

- **Qwen team** for the model and the dual-track streaming architecture: [Qwen3-TTS repo](https://github.com/QwenLM/Qwen3-TTS), [paper](https://arxiv.org/abs/2601.15621), [blog](https://qwen.ai/blog?id=qwen3tts-0115).
- **vLLM-Omni team** for the production inference stack with day-0 Qwen3-TTS support: [vllm-omni](https://github.com/vllm-project/vllm-omni).
- **Reference audio for voice cloning** is generated locally via OpenAI's TTS API. The resulting WAVs are OpenAI output; your API key stays in your shell environment.

## License

MIT. See `pyproject.toml`.
