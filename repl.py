"""Interactive REPL: type text, hear it streamed through the speakers.

Streams audio to the default output device via sounddevice as chunks arrive
from AsyncOmni — not after the whole utterance completes — so you hear the
sub-100ms TTFA behavior firsthand. Shows per-utterance TTFA and total time.

Commands (prefix ``/``):
    /list                       show the 9 CustomVoice speakers
    /speaker <name>             switch to a different speaker
    /language <name>            switch language (English, Chinese, Japanese, Korean, ...)
    /device [idx]               print or switch the audio output device
    /quit                       exit
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import time
from typing import NamedTuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import sounddevice as sd
import torch

from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

from demo import (
    DEPLOY_CONFIG_NAME,
    build_customvoice_input,
    resolve_deploy_config,
)

log = logging.getLogger("qwen3-tts-repl")


class Speaker(NamedTuple):
    name: str
    gender: str
    language: str
    description: str


SPEAKERS: list[Speaker] = [
    Speaker("Ryan",     "male",   "English",  "Dynamic male voice with rhythm."),
    Speaker("Aiden",    "male",   "English",  "Sunny American male voice."),
    Speaker("Vivian",   "female", "Chinese",  "Bright young female voice."),
    Speaker("Serena",   "female", "Chinese",  "Warm, gentle young female voice."),
    Speaker("Uncle_Fu", "male",   "Chinese",  "Seasoned male voice, mellow timbre."),
    Speaker("Dylan",    "male",   "Chinese",  "Youthful Beijing male voice."),
    Speaker("Eric",     "male",   "Chinese",  "Lively Chengdu male voice."),
    Speaker("Ono_Anna", "female", "Japanese", "Playful Japanese female voice."),
    Speaker("Sohee",    "female", "Korean",   "Warm Korean female voice."),
]
SPEAKERS_BY_NAME = {s.name.lower(): s for s in SPEAKERS}

SAMPLE_RATE = 24000  # Qwen3-TTS output rate

# Rate the OutputStream opens at. Most consumer audio stacks are
# natively 48 kHz; ALSA/PipeWire will pass this through without their
# own (often low-quality) resampler. 24→48 is a clean 2× upsample we
# can do with a high-quality polyphase filter (soxr) in Python.
# Set to 24000 to keep the Qwen rate untouched if your device is
# natively 24 kHz-friendly (uncommon).
PLAYBACK_RATE = 48000


class StreamingPlayer:
    """Persistent audio output stream fed from a thread-safe queue.

    Opens the PortAudio OutputStream once at REPL startup and leaves it
    running for the whole session. A callback (invoked by the audio
    thread) drains a ``queue.Queue`` of float32 chunks into the DAC
    output, filling with silence whenever the queue is empty. Benefits
    vs. the per-utterance ``stream.write()`` pattern:

    * No cold-start DAC warm-up per prompt (the hardware is already
      producing sound — silence — before any speech starts, so the
      first phonemes of the first word aren't eaten).
    * No per-utterance pre-buffer tax; underruns are gracefully served
      as silence instead of clicks.
    * Perceived audio latency drops to ``model_TTFA + one callback period``
      (~20 ms at the default blocksize).
    """

    def __init__(
        self,
        source_rate: int = SAMPLE_RATE,
        playback_rate: int = PLAYBACK_RATE,
        device: int | None = None,
    ) -> None:
        self._source_rate = source_rate
        self._playback_rate = playback_rate
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._pending: np.ndarray = np.zeros(0, dtype=np.float32)
        self._stream: sd.OutputStream | None = None
        self._counter_lock = threading.Lock()
        self._pushed_total = 0
        self._consumed_total = 0

        # Stateful resampler. Qwen emits 24 kHz; most audio stacks are
        # natively 48 kHz, so we upsample in Python with soxr (a VHQ
        # polyphase filter) and let the OS pass through. Prevents the
        # metallic aliasing you'd otherwise hear from ALSA/PipeWire's
        # default linear-interpolation resampler on long utterances.
        self._resampler = None
        if self._source_rate != self._playback_rate:
            import soxr

            self._resampler = soxr.ResampleStream(
                in_rate=float(self._source_rate),
                out_rate=float(self._playback_rate),
                num_channels=1,
                dtype="float32",
                quality="VHQ",
            )

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self._playback_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def set_device(self, device: int | None) -> None:
        was_running = self._stream is not None
        if was_running:
            self.stop()
        self._device = device
        if was_running:
            self.start()

    def push(self, arr: np.ndarray) -> None:
        """Enqueue one mono float32 chunk; non-blocking.

        Resamples 24 kHz → playback_rate (default 48 kHz) if needed, so
        the OS audio stack never has to resample Qwen's output with its
        own (often low-quality) resampler.
        """
        if arr.size == 0:
            return
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if self._resampler is not None:
            # soxr returns resampled samples; final flush at utterance
            # end is caller's responsibility (via `drain_resampler`).
            out = self._resampler.resample_chunk(arr)
            if out.size == 0:
                return
            arr = np.ascontiguousarray(out, dtype=np.float32)
        with self._counter_lock:
            self._pushed_total += arr.size
        # Copy so the audio thread owns its buffer even if the caller
        # frees the source array.
        self._queue.put(arr.copy())

    def drain_resampler(self) -> None:
        """Flush any samples the polyphase filter is still holding.

        Call once per utterance after the last ``push()`` so the tail
        of the audio isn't swallowed by the filter's internal delay.
        """
        if self._resampler is None:
            return
        out = self._resampler.resample_chunk(
            np.zeros(0, dtype=np.float32), last=True
        )
        if out.size == 0:
            return
        arr = np.ascontiguousarray(out, dtype=np.float32)
        with self._counter_lock:
            self._pushed_total += arr.size
        self._queue.put(arr.copy())
        # Resampler state is now consumed; create a fresh one so the
        # next utterance starts from a clean filter history.
        import soxr

        self._resampler = soxr.ResampleStream(
            in_rate=float(self._source_rate),
            out_rate=float(self._playback_rate),
            num_channels=1,
            dtype="float32",
            quality="VHQ",
        )

    def remaining(self) -> int:
        """Approximate samples still ahead of the DAC read position."""
        with self._counter_lock:
            return max(0, self._pushed_total - self._consumed_total)

    def flush_pending(self) -> None:
        """Discard any audio not yet consumed. Use when switching device."""
        with self._counter_lock:
            # Account for everything still queued/pending as consumed so
            # the counter stays monotonic.
            drained = 0
            try:
                while True:
                    drained += self._queue.get_nowait().size
            except queue.Empty:
                pass
            self._consumed_total = self._pushed_total
            self._pending = np.zeros(0, dtype=np.float32)

    def _callback(self, outdata, frames, time_info, status) -> None:
        """PortAudio callback — audio thread. Must not block or allocate heavily."""
        outdata.fill(0)
        written = 0
        if self._pending.size > 0:
            take = min(self._pending.size, frames)
            outdata[:take, 0] = self._pending[:take]
            self._pending = self._pending[take:]
            written = take
        while written < frames:
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                break
            take = min(chunk.size, frames - written)
            outdata[written:written + take, 0] = chunk[:take]
            if take < chunk.size:
                self._pending = chunk[take:]
            written += take
        if written > 0:
            with self._counter_lock:
                self._consumed_total += written


def tensor_to_mono_f32(t: torch.Tensor) -> np.ndarray:
    """Convert a 1-D audio tensor to a float32 mono numpy array for sounddevice."""
    arr = t.detach().to("cpu", dtype=torch.float32).numpy()
    return arr.reshape(-1).astype(np.float32, copy=False)


async def speak(omni, prompt_input: dict, request_id: str, player: StreamingPlayer) -> tuple[float, float, int]:
    """Stream a single utterance into the persistent StreamingPlayer.

    Returns (ttfa_ms, total_ms, total_samples). ``ttfa_ms`` is wall-clock
    from the submit call to the first chunk being enqueued. ``total_ms``
    extends until the audio thread finishes draining this utterance to
    the DAC — so successive prompts don't overlap and so bench numbers
    reflect perceived end-of-audio, not just end-of-generation.
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

    # Flush the resampler's internal history so the last few samples of
    # the utterance actually reach the DAC (polyphase filter has a
    # built-in delay that otherwise swallows the tail).
    player.drain_resampler()

    # Wait for the audio thread to drain everything this call pushed, so
    # successive prompts play in order and the reported total_ms maps to
    # when the user actually hears the end.
    while player.remaining() > 0:
        await asyncio.sleep(0.02)

    t_end = time.perf_counter()
    ttfa_ms = (t_first - t_start) * 1000.0 if t_first is not None else float("nan")
    total_ms = (t_end - t_start) * 1000.0
    return ttfa_ms, total_ms, total_samples


def print_speakers() -> None:
    print()
    print("  speaker    gender  language    description")
    for s in SPEAKERS:
        print(f"  {s.name:<10} {s.gender:<7} {s.language:<10}  {s.description}")
    print()


async def amain() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    deploy_path = resolve_deploy_config(DEPLOY_CONFIG_NAME)

    print(f"Loading Qwen3-TTS engine (one-time ~5 min)...")
    print(f"  model = {model_name}")
    print(f"  deploy = {deploy_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
    talker_cfg = getattr(cfg, "talker_config", None)

    omni = AsyncOmni(
        model=model_name,
        deploy_config=deploy_path,
        trust_remote_code=True,
        output_dir="output_audio",
    )

    # Warmup pass. Uses a longer sentence so Code2Wav captures the full
    # steady-state CUDA graph shapes, not just the first-chunk graph
    # that a one-word "hello" exercises. Without this, early user
    # utterances pay graph-capture cost mid-flight and the audio ring
    # underruns between chunks → clicks on the first prompts.
    print("Warming up CUDA graphs...")
    warm_input = build_customvoice_input(
        text=(
            "This is a warmup pass that is deliberately long enough to "
            "exercise the steady-state Code2Wav decoder path and capture "
            "every CUDA graph shape the engine will need on real requests."
        ),
        speaker="Ryan", language="English",
        tokenizer=tokenizer, talker_cfg=talker_cfg, model_name=model_name,
    )
    async for _ in omni.generate(warm_input, request_id="warmup"):
        pass
    print()
    print("Ready. Type text to speak, or `/list` for commands. `/quit` to exit.")
    print()

    current = SPEAKERS_BY_NAME["ryan"]
    current_language = current.language
    current_device: int | None = None  # default device

    # Persistent audio output — opened for the whole session so the DAC
    # never cold-starts between prompts. The callback outputs silence
    # when the queue is empty.
    player = StreamingPlayer(SAMPLE_RATE, device=current_device)
    player.start()

    utt_idx = 0
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
                print_speakers()
            elif cmd == "speaker":
                s = SPEAKERS_BY_NAME.get(arg.lower())
                if s is None:
                    print(f"  unknown speaker '{arg}'. /list for options.")
                    continue
                current = s
                current_language = s.language
                print(f"  switched to {s.name} ({s.gender}, native {s.language}): {s.description}")
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
                print(f"  unknown command '/{cmd}'. /list, /speaker, /language, /device, /quit")
            continue

        # Build the Omni request from the user's text and current speaker.
        try:
            prompt_input = build_customvoice_input(
                text=line,
                speaker=current.name,
                language=current_language,
                tokenizer=tokenizer,
                talker_cfg=talker_cfg,
                model_name=model_name,
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
        duration_s = n_samples / SAMPLE_RATE
        print(f"  TTFA={ttfa_ms:.1f}ms  total={total_ms:.1f}ms  audio={duration_s:.2f}s  (speaker={current.name})")

    player.stop()
    print("Shutting down engine...")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
