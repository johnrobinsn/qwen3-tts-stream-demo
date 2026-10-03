"""Voice-clone benchmark & sample generator.

Loads the Qwen3-TTS Base model with ``custom_voice_dir=voices`` so all the
precomputed voices sit in the SpeakerEmbeddingCache at engine startup. Then
runs N utterances through each voice and reports per-voice TTFA / total /
inter-chunk timing — because x-vectors are precomputed, per-request cost
should match the CustomVoice baseline (~47 ms TTFA on the 5090).

Output:
    samples_clone/<voice>_<n>.wav   one WAV per (voice, utterance) pair
    stdout                          summary table

Prerequisites:
    * ``voices/custom_voice_manifest.json`` + ``voices/*.safetensors``
      populated by ``precompute_voice.py`` (see README).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import statistics
from typing import NamedTuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

from demo import (
    build_base_voice_input,
    resolve_deploy_config,
    run_one,
    save_wav,
)

log = logging.getLogger("qwen3-tts-clone-bench")

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEPLOY_CONFIG = "qwen3_tts_base_voices.yaml"  # LOCAL file, not vllm_omni/deploy/
VOICES_DIR = pathlib.Path("voices")

DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank at sunset.",
    "Streaming text to speech means you hear the first word before the last one is generated.",
    "This voice was reconstructed from a short reference clip you never actually recorded.",
]


class VoiceEntry(NamedTuple):
    name: str
    mode: str                       # "xvec" or "icl"
    ref_text: str | None
    ref_code_length: int | None


def load_voice_manifest() -> list[VoiceEntry]:
    manifest_path = VOICES_DIR / "custom_voice_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run precompute_voice.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    out: list[VoiceEntry] = []
    for name, info in (manifest.get("voices") or {}).items():
        out.append(
            VoiceEntry(
                name=str(info.get("name") or name),
                mode=str(info.get("mode") or "xvec").lower(),
                ref_text=info.get("ref_text"),
                ref_code_length=info.get("ref_code_length"),
            )
        )
    return out


def _resolve_deploy() -> str:
    """Prefer a local deploy config file; fall back to shipped names."""
    local = pathlib.Path(DEPLOY_CONFIG)
    if local.exists():
        return str(local.resolve())
    return resolve_deploy_config(DEPLOY_CONFIG)


async def amain(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    voices = load_voice_manifest()
    if not voices:
        print("No precomputed voices. Run `precompute_voice.py --from-dir reference_voices` first.")
        return 2
    log.info("found %d precomputed voice(s): %s", len(voices), [v.name for v in voices])

    deploy_path = _resolve_deploy()
    log.info("deploy_config=%s model=%s", deploy_path, BASE_MODEL)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="left")
    cfg = Qwen3TTSConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    talker_cfg = getattr(cfg, "talker_config", None)

    omni = AsyncOmni(
        model=BASE_MODEL,
        deploy_config=deploy_path,
        trust_remote_code=True,
        output_dir=str(out_dir),
    )

    # Warmup with the first voice.
    v0 = voices[0]
    warm_input = build_base_voice_input(
        text="hello", voice_name=v0.name, language="English",
        tokenizer=tokenizer, talker_cfg=talker_cfg, model_name=BASE_MODEL,
        x_vector_only=(v0.mode == "xvec"),
        ref_text=v0.ref_text, ref_code_length=v0.ref_code_length,
    )
    log.info("warmup (voice=%s) ...", v0.name)
    await run_one(omni, warm_input, request_id="warmup")

    sentences = DEFAULT_SENTENCES[: args.n_runs]
    results: list[tuple[str, int, float, float, float]] = []  # (voice, idx, ttfa, total, dur)
    for v in voices:
        for i, sent in enumerate(sentences):
            inp = build_base_voice_input(
                text=sent, voice_name=v.name, language="Auto",
                tokenizer=tokenizer, talker_cfg=talker_cfg, model_name=BASE_MODEL,
                x_vector_only=(v.mode == "xvec"),
                ref_text=v.ref_text, ref_code_length=v.ref_code_length,
            )
            tl = await run_one(omni, inp, request_id=f"{v.name}_{i}")
            out_path = out_dir / f"{v.name}_{i:02d}.wav"
            save_wav(tl.audio, tl.sr, str(out_path))
            if hasattr(tl.sr, "item"):
                sr = tl.sr.item()
            elif isinstance(tl.sr, list):
                last = tl.sr[-1]
                sr = int(last.item()) if hasattr(last, "item") else int(last)
            else:
                sr = int(tl.sr) if tl.sr else 24000
            dur = sum(a.numel() for a in tl.audio) / sr
            results.append((v.name, i, tl.ttfa_ms, tl.total_ms, dur))
            log.info("  voice=%s run=%d TTFA=%.1fms total=%.1fms dur=%.2fs -> %s",
                     v.name, i, tl.ttfa_ms, tl.total_ms, dur, out_path)

    print()
    print("=" * 92)
    print(f"Qwen3-TTS voice-clone benchmark — model={BASE_MODEL}")
    print(f"  voices_dir={VOICES_DIR}  deploy={DEPLOY_CONFIG}  n_runs/voice={args.n_runs}")
    print()
    print(f"  {'voice':<14} {'runs':>4} {'TTFA p50':>10} {'TTFA min':>10} {'TTFA max':>10} {'total p50':>10}")
    for v in voices:
        rows = [r for r in results if r[0] == v.name]
        ttfas = [r[2] for r in rows]
        totals = [r[3] for r in rows]
        print(f"  {v.name:<14} {len(rows):>4} "
              f"{statistics.median(ttfas):>10.1f} {min(ttfas):>10.1f} {max(ttfas):>10.1f} "
              f"{statistics.median(totals):>10.1f}")
    all_ttfas = [r[2] for r in results]
    print()
    print(f"  overall TTFA  min={min(all_ttfas):.1f}  median={statistics.median(all_ttfas):.1f}  max={max(all_ttfas):.1f}  (ms)")
    print(f"  CustomVoice baseline (same host, warm): median ~47 ms")
    print("=" * 92)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=3, help="Utterances per voice (uses DEFAULT_SENTENCES).")
    p.add_argument("--output-dir", default="samples_clone")
    return p.parse_args()


def main() -> int:
    return asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
