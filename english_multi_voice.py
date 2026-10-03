"""Generate English samples with all plausible CustomVoice speakers.

Native English voices in the CustomVoice set are limited to Ryan + Aiden
(both male). This script runs them alongside the female speakers from
other languages, each speaking English — the multilingual model handles
cross-language synthesis but the timbre is biased toward the speaker's
native language. Useful for picking the least-accented female voice.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import NamedTuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

from demo import (
    DEPLOY_CONFIG_NAME,
    build_customvoice_input,
    resolve_deploy_config,
    run_one,
    save_wav,
)

log = logging.getLogger("qwen3-tts-english")


class Entry(NamedTuple):
    speaker: str
    gender: str
    native_lang: str
    note: str


# All 9 CustomVoice speakers made to speak English. Native English speakers
# first (Ryan/Aiden); the rest are cross-language experiments.
ENTRIES: list[Entry] = [
    Entry("Ryan",     "male",   "English", "native English — dynamic male"),
    Entry("Aiden",    "male",   "English", "native English — sunny American male"),
    Entry("Vivian",   "female", "Chinese", "bright young female — may have Chinese accent"),
    Entry("Serena",   "female", "Chinese", "warm gentle female — may have Chinese accent"),
    Entry("Ono_Anna", "female", "Japanese", "playful female — may have Japanese accent"),
    Entry("Sohee",    "female", "Korean",  "warm female — may have Korean accent"),
    Entry("Uncle_Fu", "male",   "Chinese", "seasoned male — may have Chinese accent"),
    Entry("Dylan",    "male",   "Chinese", "youthful Beijing male — strong accent likely"),
    Entry("Eric",     "male",   "Chinese", "lively Chengdu male — strong accent likely"),
]

ENGLISH_TEXT = (
    "The quick brown fox jumps over the lazy dog. This sentence is used to "
    "demonstrate the speaker's voice character and English pronunciation."
)


async def amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    out_dir = pathlib.Path("samples_english")
    out_dir.mkdir(exist_ok=True)

    deploy_path = resolve_deploy_config(DEPLOY_CONFIG_NAME)
    log.info("deploy_config=%s model=%s", deploy_path, model_name)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
    talker_cfg = getattr(cfg, "talker_config", None)

    omni = AsyncOmni(
        model=model_name,
        deploy_config=deploy_path,
        trust_remote_code=True,
        output_dir=str(out_dir),
    )

    # Warmup: Ryan (native English).
    log.info("warmup ...")
    warm = build_customvoice_input(
        text=ENGLISH_TEXT, speaker="Ryan", language="English",
        tokenizer=tokenizer, talker_cfg=talker_cfg, model_name=model_name,
    )
    await run_one(omni, warm, request_id="warmup")

    results = []
    for e in ENTRIES:
        log.info("synth speaker=%s (%s, native=%s) ...", e.speaker, e.gender, e.native_lang)
        inp = build_customvoice_input(
            text=ENGLISH_TEXT, speaker=e.speaker, language="English",
            tokenizer=tokenizer, talker_cfg=talker_cfg, model_name=model_name,
        )
        tl = await run_one(omni, inp, request_id=e.speaker)
        out_path = out_dir / f"{e.speaker}_English.wav"
        save_wav(tl.audio, tl.sr, str(out_path))

        if hasattr(tl.sr, "item"):
            sr = tl.sr.item()
        elif isinstance(tl.sr, list):
            last = tl.sr[-1]
            sr = int(last.item()) if hasattr(last, "item") else int(last)
        else:
            sr = int(tl.sr) if tl.sr else 24000
        duration_s = sum(a.numel() for a in tl.audio) / sr
        results.append((e, tl.ttfa_ms, tl.total_ms, duration_s, out_path))
        log.info("  TTFA=%.1fms total=%.1fms duration=%.2fs -> %s",
                 tl.ttfa_ms, tl.total_ms, duration_s, out_path)

    print()
    print("=" * 96)
    print(f"Qwen3-TTS English-text samples — model={model_name}")
    print(f"  text: {ENGLISH_TEXT!r}")
    print(f"  deploy: {DEPLOY_CONFIG_NAME}")
    print()
    print(f"  {'speaker':<10} {'gender':<7} {'native':<9} {'TTFA (ms)':>10} {'total (ms)':>11} {'dur (s)':>8}  note")
    for e, ttfa, tot, dur, path in results:
        print(f"  {e.speaker:<10} {e.gender:<7} {e.native_lang:<9} {ttfa:>10.1f} {tot:>11.1f} {dur:>8.2f}  {e.note}")
    print("=" * 96)
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
