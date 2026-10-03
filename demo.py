"""Streaming Qwen3-TTS demo targeting the upstream 97ms TTFA claim.

Runs vLLM-Omni's AsyncOmni with the fused single-GPU single-stage profile
(``qwen3_tts_fused_single_gpu.yaml``). Measures time-to-first-audio by
timestamping each chunk yielded by ``AsyncOmni.generate``.

Target hardware is the RTX 5090 at ``CUDA_VISIBLE_DEVICES=1`` on this host.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import statistics
import time
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import soundfile as sf
import torch

from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
)

log = logging.getLogger("qwen3-tts-stream-demo")

# Shipped single-GPU low-latency profile: Model Runner V2 + Talker + Code2Wav
# with shared-memory connector, `talker_first_audio: true`, and
# `initial_codec_chunk_frames: 1` for an aggressive first chunk.
#
# NOTE: the single-stage "fused" variant (``qwen3_tts_fused_single_gpu.yaml``)
# that avoids the inter-stage SHM handoff landed on vllm-omni main on
# 2026-10-02 and is NOT in the 0.30.0 PyPI wheel. If you install vllm-omni
# from source after that commit, switch this to the fused profile for
# lower TTFA.
DEPLOY_CONFIG_NAME = "qwen3_tts_mrv2.yaml"


def resolve_deploy_config(name: str) -> str:
    """Return absolute path to a shipped vllm-omni deploy YAML by filename."""
    import vllm_omni

    base = pathlib.Path(vllm_omni.__file__).parent / "deploy"
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"deploy config {name} not found under {base}")
    return str(path)


def estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
    tokenizer,
    talker_cfg,
) -> int:
    """Length-only placeholder for ``prompt_token_ids``.

    The Talker replaces input embeddings via its ``preprocess`` hook, so values
    don't matter — only the length must match what ``preprocess`` emits.
    """
    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=additional_information,
        task_type=additional_information["task_type"][0],
        tokenize_prompt=lambda t: tokenizer(t, padding=False)["input_ids"],
        codec_language_id=getattr(talker_cfg, "codec_language_id", None),
        spk_is_dialect=getattr(talker_cfg, "spk_is_dialect", None),
        estimate_ref_code_len=lambda _ref: None,
    )


def build_customvoice_input(
    text: str,
    speaker: str,
    language: str,
    tokenizer,
    talker_cfg,
    model_name: str,
) -> dict:
    """Build an Omni request for the CustomVoice task (predefined speaker)."""
    additional_information = {
        "task_type": ["CustomVoice"],
        "text": [text],
        "language": [language],
        "speaker": [speaker],
        "instruct": [""],
        "max_new_tokens": [2048],
        # False = emit Code2Wav chunks incrementally through the SHM
        # connector (what we want for TTFA). True would defer all audio
        # until the Talker completes (good for throughput, not latency).
        # See vllm_omni/model_executor/stage_input_processors/qwen3_tts.py
        # for the gating logic.
        "full_utterance_decode": [False],
    }
    plen = estimate_prompt_len(additional_information, model_name, tokenizer, talker_cfg)
    return {
        "prompt_token_ids": [0] * plen,
        "additional_information": additional_information,
    }


def build_base_voice_input(
    text: str,
    voice_name: str,
    language: str,
    tokenizer,
    talker_cfg,
    model_name: str,
    *,
    x_vector_only: bool = False,
    ref_text: str | None = None,
    ref_code_length: int | None = None,
) -> dict:
    """Build an Omni request for the Base task using a *precomputed* voice.

    Requires the engine to have been started with a deploy config that sets
    ``custom_voice_dir`` so ``voice_name`` resolves against the preloaded
    SpeakerEmbeddingCache (no per-request codec encoding).

    Args:
        voice_name: Must match the ``name`` field in the manifest (and the
            safetensors file stem).
        x_vector_only: ``True`` → use only the x-vector (xvec mode;
            precompute must have been run with ``--mode xvec``).
            ``False`` → ICL mode: use x-vector + ref_code (precompute
            must have been run with ``--mode icl``, which also stashed
            ``ref_text`` in the manifest).
        ref_text / ref_code_length: Only needed for prompt-length estimation
            under ICL mode. Read these from
            ``voices/custom_voice_manifest.json``.
    """
    additional_information = {
        "task_type": ["Base"],
        "text": [text],
        "language": [language],
        # The input processor keys the SpeakerEmbeddingCache lookup off
        # ``speaker[0]`` lowercased + the current xvec/icl namespace, so this
        # must match the manifest ``name`` (lower). See
        # qwen3_tts/prompt_embeds_builder.py:1197-1228.
        "speaker": [voice_name],
        "x_vector_only_mode": [bool(x_vector_only)],
        "max_new_tokens": [2048],
        "full_utterance_decode": [False],
    }
    # Prompt-length estimation under ICL mode needs the ref_text and the
    # ref_code frame count. We bypass the codec-encoder path (precomputed!)
    # by returning the stashed length directly.
    if (not x_vector_only) and ref_text:
        additional_information["ref_text"] = [ref_text]
    plen = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=additional_information,
        task_type="Base",
        tokenize_prompt=lambda t: tokenizer(t, padding=False)["input_ids"],
        codec_language_id=getattr(talker_cfg, "codec_language_id", None),
        spk_is_dialect=getattr(talker_cfg, "spk_is_dialect", None),
        estimate_ref_code_len=lambda _ref: (ref_code_length if ref_code_length else None),
    )
    return {
        "prompt_token_ids": [0] * plen,
        "additional_information": additional_information,
    }


def save_wav(audio_list, sr_raw, out_path: str) -> None:
    """Concatenate audio chunks and write a 16-bit PCM wav."""
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
    audio_tensor = torch.cat(audio_list, dim=-1) if isinstance(audio_list, list) else audio_list
    sf.write(out_path, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")


class ChunkTimeline:
    """Captures per-chunk wall-clock from the start of ``generate``."""

    def __init__(self) -> None:
        self.t_start: float = 0.0
        self.chunk_ts: list[float] = []
        self.chunk_sample_counts: list[int] = []
        self.finished_at: float | None = None
        self.audio: list[torch.Tensor] = []
        self.sr: Any = None

    @property
    def ttfa_ms(self) -> float:
        if not self.chunk_ts:
            return float("nan")
        return (self.chunk_ts[0] - self.t_start) * 1000.0

    @property
    def total_ms(self) -> float:
        if self.finished_at is None:
            return float("nan")
        return (self.finished_at - self.t_start) * 1000.0

    @property
    def inter_chunk_ms(self) -> list[float]:
        return [(b - a) * 1000.0 for a, b in zip(self.chunk_ts[:-1], self.chunk_ts[1:])]

    def summary(self) -> str:
        parts = [
            f"TTFA={self.ttfa_ms:.1f}ms",
            f"chunks={len(self.chunk_ts)}",
            f"total={self.total_ms:.1f}ms",
        ]
        if self.inter_chunk_ms:
            parts.append(f"inter_chunk_p50={statistics.median(self.inter_chunk_ms):.1f}ms")
            parts.append(f"inter_chunk_max={max(self.inter_chunk_ms):.1f}ms")
        return " ".join(parts)


async def run_one(omni, prompt_input: dict, request_id: str, debug: bool = False) -> ChunkTimeline:
    tl = ChunkTimeline()
    tl.t_start = time.perf_counter()
    iter_idx = 0
    async for stage_output in omni.generate(prompt_input, request_id=request_id):
        now = time.perf_counter()
        out0 = stage_output.outputs[0]
        mm = getattr(out0, "multimodal_output", None) or {}
        finished = stage_output.finished

        if debug:
            try:
                mm_keys = list(mm.keys()) if hasattr(mm, "keys") else "no-keys"
            except Exception as e:
                mm_keys = f"keys-error:{e}"
            audio = None
            if hasattr(mm, "get"):
                audio = mm.get("audio")
            def _audio_summary(a):
                if a is None:
                    return "none"
                if isinstance(a, list):
                    if not a:
                        return "list[0]"
                    first = a[0]
                    return f"list[{len(a)}] first.shape={tuple(first.shape) if hasattr(first, 'shape') else type(first).__name__}"
                if hasattr(a, "shape"):
                    return f"tensor{tuple(a.shape)}"
                return type(a).__name__
            log.info(
                "  iter=%d t=%.3fs finished=%s stage_id=%s mm_type=%s mm_keys=%s audio=%s",
                iter_idx,
                now - tl.t_start,
                finished,
                getattr(stage_output, "stage_id", "?"),
                type(mm).__name__,
                mm_keys,
                _audio_summary(audio),
            )
        iter_idx += 1

        # MultimodalPayload is a Mapping, not a plain dict — rely on .get()
        # alone, not isinstance(mm, dict).
        audio = mm.get("audio") if hasattr(mm, "get") else None
        if tl.sr is None and hasattr(mm, "get"):
            tl.sr = mm.get("sr")

        if not finished:
            if audio is None:
                continue
            tl.chunk_ts.append(now)
            if isinstance(audio, list):
                for a in audio:
                    tl.audio.append(a)
                tl.chunk_sample_counts.append(sum(int(a.numel()) for a in audio))
            else:
                tl.audio.append(audio)
                tl.chunk_sample_counts.append(int(audio.numel()))
        else:
            tl.finished_at = now
            if audio is not None:
                # Append the finishing tail; we keep the per-chunk history
                # intact (chunk_ts is unchanged — the final boundary is
                # recorded as `finished_at`).
                if isinstance(audio, list):
                    tl.audio.extend(audio)
                else:
                    tl.audio.append(audio)
    return tl


async def amain(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deploy_path = resolve_deploy_config(DEPLOY_CONFIG_NAME)
    log.info("deploy_config=%s", deploy_path)
    log.info("model=%s", args.model)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    cfg = Qwen3TTSConfig.from_pretrained(args.model, trust_remote_code=True)
    talker_cfg = getattr(cfg, "talker_config", None)

    omni = AsyncOmni(
        model=args.model,
        deploy_config=deploy_path,
        trust_remote_code=True,
        output_dir=str(out_dir),
    )

    prompt_input = build_customvoice_input(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        tokenizer=tokenizer,
        talker_cfg=talker_cfg,
        model_name=args.model,
    )

    # ------------------------------------------------------------------
    # Warmup pass. First inference pays CUDA-graph capture and allocator
    # warmup; the 97ms number cannot be interpreted against a cold run.
    # ------------------------------------------------------------------
    log.info("warmup run (not counted) ...")
    warmup_tl = await run_one(omni, prompt_input, request_id="warmup", debug=args.debug)
    log.info("warmup done: %s", warmup_tl.summary())
    save_wav(warmup_tl.audio, warmup_tl.sr, str(out_dir / "warmup.wav"))

    # ------------------------------------------------------------------
    # Measurement passes.
    # ------------------------------------------------------------------
    timelines: list[ChunkTimeline] = []
    for i in range(args.n_runs):
        log.info("run %d/%d ...", i + 1, args.n_runs)
        tl = await run_one(omni, prompt_input, request_id=f"run{i}", debug=args.debug)
        log.info("run %d: %s", i + 1, tl.summary())
        save_wav(tl.audio, tl.sr, str(out_dir / f"run_{i:02d}.wav"))
        timelines.append(tl)

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    ttfas = [tl.ttfa_ms for tl in timelines]
    totals = [tl.total_ms for tl in timelines]
    print()
    print("=" * 72)
    print(f"Qwen3-TTS streaming demo — model={args.model}")
    print(f"  text: {args.text!r}  speaker={args.speaker} language={args.language}")
    print(f"  deploy: {DEPLOY_CONFIG_NAME}  n_runs={args.n_runs}")
    print()
    print(f"  TTFA  (ms)  min={min(ttfas):.1f}  median={statistics.median(ttfas):.1f}  max={max(ttfas):.1f}")
    print(f"  total (ms)  min={min(totals):.1f}  median={statistics.median(totals):.1f}  max={max(totals):.1f}")
    print(f"  Qwen upstream claim: 97 ms TTFA (unspecified GPU; dual-track pipeline)")
    print("=" * 72)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="HF repo id. The 0.6B-CustomVoice variant minimizes first-chunk latency "
             "(fewest params, no ref-audio tokenization).",
    )
    p.add_argument("--text", default="Hello, this is a streaming time-to-first-audio test.")
    p.add_argument("--speaker", default="Ryan", help="CustomVoice predefined speaker.")
    p.add_argument("--language", default="English")
    p.add_argument("--output-dir", default="output_audio")
    p.add_argument("--n-runs", type=int, default=3, help="Number of measurement runs after warmup.")
    p.add_argument("--debug", action="store_true", help="Log every yield from AsyncOmni.generate.")
    return p.parse_args()


def main() -> int:
    return asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
