"""Generate samples with a mix of predefined CustomVoice speakers.

Reuses the AsyncOmni instance so the ~5 min engine init is paid once. Each
speaker is run with text in its native language per the model card
recommendation ("We recommend using each speaker's native language for the
best results.").

Outputs land in ``samples/<speaker>_<language>.wav`` (24 kHz PCM) with a
summary printed to stdout.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
from typing import NamedTuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import soundfile as sf
import torch

from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

from demo import (
    DEPLOY_CONFIG_NAME,
    build_customvoice_input,
    resolve_deploy_config,
    run_one,
    save_wav,
)

log = logging.getLogger("qwen3-tts-multi-voice")


class VoiceSample(NamedTuple):
    speaker: str
    gender: str
    language: str
    text: str


# Mix of male and female voices across 4 languages. Each text is a short
# greeting / self-introduction in the speaker's native language.
VOICE_SAMPLES: list[VoiceSample] = [
    VoiceSample("Ryan", "male", "English",
                "Hello there! I'm Ryan, a dynamic English voice from Qwen three TTS."),
    VoiceSample("Aiden", "male", "English",
                "Hi, I'm Aiden, a sunny American voice. This is a streaming demo."),
    VoiceSample("Vivian", "female", "Chinese",
                "大家好，我是薇薇安，一个明亮的年轻女声。"),
    VoiceSample("Serena", "female", "Chinese",
                "你好，我是赛琳娜，温柔的女生，很高兴认识你。"),
    VoiceSample("Uncle_Fu", "male", "Chinese",
                "大家好，我是傅叔，一个成熟稳重的男声。"),
    VoiceSample("Ono_Anna", "female", "Japanese",
                "こんにちは、小野アンナです。日本語の音声サンプルです。"),
    VoiceSample("Sohee", "female", "Korean",
                "안녕하세요, 저는 소희입니다. 한국어 음성 샘플입니다."),
]


async def amain() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    out_dir = pathlib.Path("samples")
    out_dir.mkdir(exist_ok=True)

    deploy_path = resolve_deploy_config(DEPLOY_CONFIG_NAME)
    log.info("deploy_config=%s", deploy_path)
    log.info("model=%s", model_name)

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

    # ------------------------------------------------------------------
    # Warmup pass (not reported). The first inference pays CUDA graph
    # specialization regardless of which speaker is used.
    # ------------------------------------------------------------------
    warm = VOICE_SAMPLES[0]
    log.info("warmup with speaker=%s ...", warm.speaker)
    warmup_input = build_customvoice_input(
        text=warm.text,
        speaker=warm.speaker,
        language=warm.language,
        tokenizer=tokenizer,
        talker_cfg=talker_cfg,
        model_name=model_name,
    )
    await run_one(omni, warmup_input, request_id="warmup")

    # ------------------------------------------------------------------
    # Per-speaker runs.
    # ------------------------------------------------------------------
    results = []
    for sample in VOICE_SAMPLES:
        log.info("synthesizing speaker=%s (%s %s) ...", sample.speaker, sample.gender, sample.language)
        prompt_input = build_customvoice_input(
            text=sample.text,
            speaker=sample.speaker,
            language=sample.language,
            tokenizer=tokenizer,
            talker_cfg=talker_cfg,
            model_name=model_name,
        )
        tl = await run_one(omni, prompt_input, request_id=f"{sample.speaker}")
        out_path = out_dir / f"{sample.speaker}_{sample.language}.wav"
        save_wav(tl.audio, tl.sr, str(out_path))
        duration_s = sum(a.numel() for a in tl.audio) / (tl.sr if isinstance(tl.sr, int) else 24000)
        if hasattr(tl.sr, "item"):
            sr = tl.sr.item()
        elif isinstance(tl.sr, list):
            sr = int(tl.sr[-1].item()) if hasattr(tl.sr[-1], "item") else int(tl.sr[-1])
        else:
            sr = int(tl.sr) if tl.sr else 24000
        duration_s = sum(a.numel() for a in tl.audio) / sr
        results.append(
            {
                "speaker": sample.speaker,
                "gender": sample.gender,
                "language": sample.language,
                "ttfa_ms": tl.ttfa_ms,
                "total_ms": tl.total_ms,
                "duration_s": duration_s,
                "file": str(out_path),
            }
        )
        log.info("  TTFA=%.1fms total=%.1fms duration=%.2fs -> %s",
                 tl.ttfa_ms, tl.total_ms, duration_s, out_path)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print()
    print("=" * 86)
    print(f"Qwen3-TTS multi-voice samples — model={model_name}")
    print(f"  deploy: {DEPLOY_CONFIG_NAME}")
    print()
    print(f"  {'speaker':<10} {'gender':<7} {'language':<9} {'TTFA (ms)':>10} {'total (ms)':>11} {'dur (s)':>8}  file")
    for r in results:
        print(f"  {r['speaker']:<10} {r['gender']:<7} {r['language']:<9} {r['ttfa_ms']:>10.1f} {r['total_ms']:>11.1f} {r['duration_s']:>8.2f}  {r['file']}")
    print("=" * 86)
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
