"""Interactive REPL for Qwen3-TTS Base-model voice cloning.

Loads the Base model with ``custom_voice_dir=voices`` so every precomputed
voice sits in the engine's SpeakerEmbeddingCache. Type text, hear it in the
selected clone, measure TTFA live.

Commands:
    /list                       show precomputed voices from the manifest
    /voice <name>               switch to a precomputed voice
    /language <name>            set synthesis language (Auto, English, Chinese, ...)
    /device [idx]               list output devices or switch
    /quit                       exit

To add a new voice: ``/quit``, drop a WAV + matching .txt into
``reference_voices/``, run ``./precompute.sh`` (or
``.venv/bin/python precompute_voice.py --from-dir reference_voices``),
then launch this script again. Runtime cloning without engine restart
would need SpeakerEmbeddingCache IPC, which the offline AsyncOmni path
doesn't currently expose.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
from typing import NamedTuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import sounddevice as sd
import torch

from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

from demo import build_base_voice_input, resolve_deploy_config
from repl import SAMPLE_RATE, StreamingPlayer, tensor_to_mono_f32

log = logging.getLogger("qwen3-tts-clone-repl")

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEPLOY_CONFIG = "qwen3_tts_base_voices.yaml"
VOICES_DIR = pathlib.Path("voices")


class CloneVoice(NamedTuple):
    name: str
    mode: str
    ref_text: str | None
    ref_code_length: int | None


def load_voices() -> dict[str, CloneVoice]:
    manifest_path = VOICES_DIR / "custom_voice_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    out: dict[str, CloneVoice] = {}
    for name, info in (manifest.get("voices") or {}).items():
        v = CloneVoice(
            name=str(info.get("name") or name),
            mode=str(info.get("mode") or "xvec").lower(),
            ref_text=info.get("ref_text"),
            ref_code_length=info.get("ref_code_length"),
        )
        out[v.name.lower()] = v
    return out


async def speak(omni, prompt_input: dict, request_id: str, player: StreamingPlayer):
    """Push one utterance into the persistent StreamingPlayer.

    TTFA is first-chunk arrival. total_ms waits for the audio thread to
    drain this utterance so successive prompts don't overlap.
    """
    t_start = time.perf_counter()
    t_first: float | None = None
    total_samples = 0

    async for stage_output in omni.generate(prompt_input, request_id=request_id):
        mm = stage_output.outputs[0].multimodal_output
        if mm is None:
            continue
        audio = mm.get("audio") if hasattr(mm, "get") else None
        if audio is None:
            continue
        chunks = audio if isinstance(audio, list) else [audio]
        for c in chunks:
            arr = tensor_to_mono_f32(c)
            if arr.size == 0:
                continue
            if t_first is None:
                t_first = time.perf_counter()
            total_samples += arr.size
            player.push(arr)

    # Flush the resampler's internal history so the utterance tail
    # reaches the DAC (polyphase filter has a built-in delay).
    player.drain_resampler()

    while player.remaining() > 0:
        await asyncio.sleep(0.02)

    t_end = time.perf_counter()
    ttfa_ms = (t_first - t_start) * 1000.0 if t_first is not None else float("nan")
    total_ms = (t_end - t_start) * 1000.0
    return ttfa_ms, total_ms, total_samples


def _resolve_deploy() -> str:
    local = pathlib.Path(DEPLOY_CONFIG)
    if local.exists():
        return str(local.resolve())
    return resolve_deploy_config(DEPLOY_CONFIG)


async def amain() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    voices = load_voices()
    if not voices:
        print(f"ERROR: no precomputed voices under {VOICES_DIR}/.")
        print(f"       Run precompute_voice.py --from-dir reference_voices first.")
        return 2

    deploy_path = _resolve_deploy()
    print(f"Loading Qwen3-TTS Base engine (one-time ~5 min)...")
    print(f"  model  = {BASE_MODEL}")
    print(f"  deploy = {deploy_path}")
    print(f"  voices = {len(voices)}: {', '.join(v.name for v in voices.values())}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="left")
    cfg = Qwen3TTSConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    talker_cfg = getattr(cfg, "talker_config", None)

    omni = AsyncOmni(
        model=BASE_MODEL,
        deploy_config=deploy_path,
        trust_remote_code=True,
        output_dir="output_audio",
    )

    # Warmup pass — no playback. Uses a longer sentence so Code2Wav
    # captures CUDA graphs for the full steady-state chunk size, not just
    # the tiny first-chunk graph that a one-word "hello" would exercise.
    # Without this, the first few *real* utterances pay graph-capture
    # cost mid-flight → the Code2Wav output stalls between chunks →
    # audio ring underruns → clicks on the first couple of prompts.
    current = next(iter(voices.values()))
    print(f"Warming up CUDA graphs with voice={current.name}...")
    warm = build_base_voice_input(
        text=(
            "This is a warmup pass that is deliberately long enough to "
            "exercise the steady-state Code2Wav decoder path and capture "
            "every CUDA graph shape the engine will need on real requests."
        ),
        voice_name=current.name, language="English",
        tokenizer=tokenizer, talker_cfg=talker_cfg, model_name=BASE_MODEL,
        x_vector_only=(current.mode == "xvec"),
        ref_text=current.ref_text, ref_code_length=current.ref_code_length,
    )
    async for _ in omni.generate(warm, request_id="warmup"):
        pass
    print()
    print("Ready. Type text to speak, `/list` for commands, `/quit` to exit.")
    print()

    current_language = "English"
    current_device: int | None = None
    utt_idx = 0

    # Persistent audio output — opened once, kept warm across prompts.
    player = StreamingPlayer(SAMPLE_RATE, device=current_device)
    player.start()

    while True:
        try:
            prompt = f"[{current.name}/{current_language}]> "
            line = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in {"quit", "exit", "q"}:
                break
            elif cmd == "list":
                print()
                print(f"  {'voice':<16} {'mode':<5} {'ref_frames':>10}  ref_text (truncated)")
                for v in voices.values():
                    rt = (v.ref_text or "")[:60].replace("\n", " ")
                    print(f"  {v.name:<16} {v.mode:<5} {str(v.ref_code_length or '-'):>10}  {rt}")
                print()
            elif cmd == "voice":
                nxt = voices.get(arg.lower())
                if nxt is None:
                    print(f"  unknown voice '{arg}'. /list for options.")
                    continue
                current = nxt
                print(f"  switched to {current.name} (mode={current.mode})")
            elif cmd == "language":
                if not arg:
                    print(f"  current language: {current_language}")
                    continue
                current_language = arg
                print(f"  language set to {current_language}")
            elif cmd == "device":
                if arg == "":
                    print(f"  current device: {current_device} (None = system default)")
                    print()
                    for i, d in enumerate(sd.query_devices()):
                        if d.get("max_output_channels", 0) > 0:
                            print(f"    [{i}] {d['name']}  out_ch={d['max_output_channels']}")
                    continue
                try:
                    current_device = int(arg)
                    player.set_device(current_device)
                    print(f"  device set to [{current_device}]")
                except ValueError:
                    print(f"  not a device index: {arg}")
            else:
                print(f"  unknown command '/{cmd}'. Available: /list /voice /language /device /quit")
            continue

        try:
            prompt_input = build_base_voice_input(
                text=line,
                voice_name=current.name,
                language=current_language,
                tokenizer=tokenizer,
                talker_cfg=talker_cfg,
                model_name=BASE_MODEL,
                x_vector_only=(current.mode == "xvec"),
                ref_text=current.ref_text,
                ref_code_length=current.ref_code_length,
            )
        except Exception as e:
            print(f"  input build failed: {e}")
            continue
        try:
            ttfa_ms, total_ms, n_samples = await speak(
                omni, prompt_input, request_id=f"utt{utt_idx}", player=player
            )
        except Exception as e:
            print(f"  synthesis failed: {e}")
            continue
        utt_idx += 1
        dur_s = n_samples / SAMPLE_RATE
        print(f"  TTFA={ttfa_ms:.1f}ms  total={total_ms:.1f}ms  audio={dur_s:.2f}s  (voice={current.name})")

    player.stop()
    print("Shutting down engine...")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
