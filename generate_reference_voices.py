"""Generate reference audio clips using OpenAI TTS for Qwen3-TTS voice cloning.

Produces (``reference_voices/<name>.wav``, ``reference_voices/<name>.txt``)
pairs — the WAV goes into the Qwen3-TTS Base model's codec encoder to extract
a speaker x-vector; the TXT is the transcript (required for ICL mode).

Idempotent: skips any voice whose WAV + TXT already exist.

Security:
    OPENAI_API_KEY is read from the environment only — never written to disk,
    the manifest, or stdout. Set it before running:

        export OPENAI_API_KEY="sk-..."
        .venv/bin/python generate_reference_voices.py

Output files are safe to commit (they're OpenAI TTS output, not your key).
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import NamedTuple


class VoicePrompt(NamedTuple):
    name: str          # local id used by Qwen3-TTS precompute (/voices/<name>.safetensors)
    openai_voice: str  # OpenAI TTS voice identifier (alloy, echo, nova, ...)
    gender: str        # male / female (for display only; not used by Qwen)
    text: str          # transcript — authored here so we can save it with the audio


# Four reference clips — two male, two female. Each ~5–8 s of English.
# Transcripts are authored here so we can save them alongside the audio
# (the Base ICL path needs ref_text matching the audio word-for-word).
VOICES: list[VoicePrompt] = [
    VoicePrompt(
        name="oai_echo",
        openai_voice="echo",
        gender="male",
        text="Hi, I'm Echo. My voice is clear and even, suited for narration and tutorials.",
    ),
    VoicePrompt(
        name="oai_onyx",
        openai_voice="onyx",
        gender="male",
        text="Hello there. I'm Onyx, with a deeper and more resonant tone that carries authority.",
    ),
    VoicePrompt(
        name="oai_nova",
        openai_voice="nova",
        gender="female",
        text="Hi, I'm Nova. My voice is warm and energetic, great for conversational assistants.",
    ),
    VoicePrompt(
        name="oai_shimmer",
        openai_voice="shimmer",
        gender="female",
        text="Hello. I'm Shimmer, with a soft and friendly tone that's easy to listen to for long sessions.",
    ),
]


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. export it, then rerun.", file=sys.stderr)
        print("  export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(__file__).parent / "reference_voices"
    out_dir.mkdir(exist_ok=True)

    from openai import OpenAI

    # Client reads OPENAI_API_KEY from env — don't pass it explicitly so it
    # can't accidentally end up in a traceback.
    client = OpenAI()

    for v in VOICES:
        wav_path = out_dir / f"{v.name}.wav"
        txt_path = out_dir / f"{v.name}.txt"
        if wav_path.exists() and txt_path.exists():
            print(f"skip  {v.name} ({v.gender}, {v.openai_voice}) — already exists")
            continue
        print(f"gen   {v.name} ({v.gender}, {v.openai_voice}) ...", end=" ", flush=True)
        try:
            # gpt-4o-mini-tts is the current higher-quality model; falls back
            # to tts-1-hd if the account doesn't have access.
            try:
                resp = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=v.openai_voice,
                    input=v.text,
                    response_format="wav",
                )
            except Exception:
                resp = client.audio.speech.create(
                    model="tts-1-hd",
                    voice=v.openai_voice,
                    input=v.text,
                    response_format="wav",
                )
            wav_bytes = resp.content
        except Exception as exc:
            # Keep the error message minimal — don't echo request details that
            # might include auth headers from a wrapped debug object.
            print(f"FAILED: {type(exc).__name__}")
            continue
        wav_path.write_bytes(wav_bytes)
        txt_path.write_text(v.text + "\n", encoding="utf-8")
        print(f"wrote {wav_path.name} ({len(wav_bytes)} bytes)")

    # Index — useful for scripts that iterate the voice set without
    # importing this file.
    index_path = out_dir / "index.txt"
    with index_path.open("w", encoding="utf-8") as f:
        f.write("# name\topenai_voice\tgender\n")
        for v in VOICES:
            if (out_dir / f"{v.name}.wav").exists():
                f.write(f"{v.name}\t{v.openai_voice}\t{v.gender}\n")
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
