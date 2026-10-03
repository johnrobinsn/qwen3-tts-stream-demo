"""Precompute a Qwen3-TTS Base-model voice profile from a reference audio clip.

Takes a ref audio (local path or http(s) URL) + its transcript, extracts the
speaker x-vector (and optionally the ref_code for ICL mode) using the Base
model's speaker encoder + speech tokenizer, and writes
``voices/<name>.safetensors`` + a shared ``voices/custom_voice_manifest.json``.

The resulting directory is pointed at by ``custom_voice_dir`` in the deploy
config so the engine loads every voice at startup into its SpeakerEmbeddingCache;
request-time cost is then a hash-table lookup — no per-request codec encoding.

Thin wrapper around vllm-omni's upstream ``precompute_custom_voice.py`` with:
    * URL ref-audio support (downloads to /tmp if needed),
    * a batch mode that pre-processes every clip in ``reference_voices/``.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
import urllib.parse
import urllib.request

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def _materialize_ref(ref: str) -> str:
    """Return a local path for ``ref``. Downloads http(s) URLs to a temp file."""
    parsed = urllib.parse.urlparse(ref)
    if parsed.scheme in ("http", "https"):
        suffix = os.path.splitext(parsed.path)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            with urllib.request.urlopen(ref) as resp:
                tmp.write(resp.read())
            return tmp.name
    return ref


def precompute_single(
    *,
    name: str,
    ref_audio: str,
    ref_text: str | None,
    mode: str,
    output_dir: pathlib.Path,
    device: str,
) -> None:
    """Run the upstream precompute function on a single voice."""
    # Lazy-import so just running --help doesn't pull torch.
    import torch

    # Avoid depending on vllm-omni's example path layout — reimplement the
    # core _write_voice logic locally, since the upstream script resolves
    # paths relative to the repo checkout.
    from safetensors.torch import save_file
    import numpy as np
    import json

    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSSpeakerEncoder
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import mel_spectrogram
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    from vllm_omni.utils.custom_voice_io import safe_voice_stem

    import soundfile as sf
    from transformers.utils.hub import cached_file

    local_ref = _materialize_ref(ref_audio)
    try:
        wav, sr = sf.read(local_ref, dtype="float32")
    finally:
        if local_ref != ref_audio and os.path.exists(local_ref):
            try:
                os.unlink(local_ref)
            except OSError:
                pass

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < 1024:
        raise ValueError(f"ref audio too short: {wav.size} samples")

    # Resolve the Base model checkpoint directory.
    cfg_path = cached_file(BASE_MODEL, "config.json")
    model_dir = os.path.dirname(cfg_path)

    config = Qwen3TTSConfig.from_pretrained(model_dir)
    dev = torch.device(device)

    # ---- Speaker encoder → x-vector ----
    encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    from safetensors import safe_open

    speaker_state: dict = {}
    for shard in sorted(pathlib.Path(model_dir).glob("model*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("speaker_encoder."):
                    speaker_state[key.removeprefix("speaker_encoder.")] = f.get_tensor(key)
    if not speaker_state:
        raise RuntimeError(f"No speaker_encoder.* weights in {model_dir}")
    encoder.load_state_dict(speaker_state)
    encoder.to(device=dev, dtype=torch.bfloat16).eval()

    SR, N_MELS, N_FFT, HOP, WIN, FMIN, FMAX = 24000, 128, 1024, 256, 1024, 0, 12000
    if sr != SR:
        from vllm.multimodal.audio import AudioResampler

        wav = AudioResampler(target_sr=SR).resample(wav, orig_sr=int(sr))
        sr = SR
    wav_t = torch.from_numpy(wav).to(device=dev, dtype=torch.float32)
    mels = mel_spectrogram(
        wav_t.unsqueeze(0),
        n_fft=N_FFT, num_mels=N_MELS, sampling_rate=SR,
        hop_size=HOP, win_size=WIN, fmin=FMIN, fmax=FMAX,
    ).transpose(1, 2)
    with torch.inference_mode():
        speaker_embedding = encoder(mels.to(device=dev, dtype=torch.bfloat16))[0].float().cpu().contiguous()

    tensors = {"speaker_embedding": speaker_embedding}

    # ---- Speech tokenizer → ref_code (ICL mode only) ----
    if mode == "icl":
        if not ref_text or not ref_text.strip():
            raise ValueError("--mode icl requires --ref-text (transcript of ref audio)")
        tok = Qwen3TTSTokenizer.from_pretrained(
            str(pathlib.Path(model_dir) / "speech_tokenizer"),
            torch_dtype=torch.bfloat16,
        )
        try:
            del tok.model.decoder
            tok.model.decoder = None
            tok.model.encoder.to(dev)
            tok.device = dev
        except Exception:
            tok.device = dev
        with torch.inference_mode():
            enc = tok.encode(wav, sr=int(sr), return_dict=True)
        codes = getattr(enc, "audio_codes", None)
        if isinstance(codes, list):
            codes = codes[0] if codes else None
        if not isinstance(codes, torch.Tensor):
            raise RuntimeError("speech_tokenizer did not return audio_codes")
        if codes.ndim == 3:
            codes = codes[0]
        tensors["ref_code"] = codes.to(dtype=torch.int32, device="cpu").contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_voice_stem(name)}.safetensors"
    save_file(tensors, str(output_dir / filename))

    hidden_size = int(getattr(config.talker_config, "hidden_size", speaker_embedding.numel()))
    manifest_path = output_dir / "custom_voice_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "model_type": "qwen3_tts",
            "model": BASE_MODEL,
            "hidden_size": hidden_size,
            "voices": {},
        }
    entry: dict = {
        "name": name,
        "file": filename,
        "mode": mode,
        "embedding_dim": int(speaker_embedding.numel()),
    }
    if "ref_code" in tensors:
        entry["ref_code_length"] = int(tensors["ref_code"].shape[0])
    if ref_text:
        entry["ref_text"] = ref_text
    manifest.setdefault("voices", {})[name] = entry
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {output_dir / filename}  (mode={mode}, dim={speaker_embedding.numel()}, "
          f"ref_frames={entry.get('ref_code_length', '-')})")


def batch_from_reference_dir(voices_dir: pathlib.Path, ref_dir: pathlib.Path, mode: str, device: str) -> int:
    """Precompute every (wav, txt) pair in ``ref_dir``. Skips existing voices."""
    manifest_path = voices_dir / "custom_voice_manifest.json"
    existing: set[str] = set()
    if manifest_path.exists():
        import json

        existing = set((json.loads(manifest_path.read_text()).get("voices") or {}).keys())
    pairs: list[tuple[str, pathlib.Path, pathlib.Path]] = []
    for wav in sorted(ref_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"skip  {wav.name}: no matching .txt")
            continue
        name = wav.stem
        if name in existing:
            print(f"skip  {name}: already in manifest")
            continue
        pairs.append((name, wav, txt))

    if not pairs:
        print(f"no new voices to precompute under {ref_dir}")
        return 0

    for name, wav, txt in pairs:
        ref_text = txt.read_text(encoding="utf-8").strip()
        print(f"precompute {name} <- {wav.name}  ({len(ref_text)} chars ref_text)")
        precompute_single(
            name=name, ref_audio=str(wav), ref_text=ref_text,
            mode=mode, output_dir=voices_dir, device=device,
        )
    return len(pairs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["xvec", "icl"], default="icl",
                   help="xvec = speaker embedding only (fastest TTFA); "
                        "icl = speaker embedding + ref_code (fuller fidelity)")
    p.add_argument("--output-dir", default="voices",
                   help="Where to write <name>.safetensors + custom_voice_manifest.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", help="Voice name (filename stem). Required unless --from-dir.")
    p.add_argument("--ref-audio", help="Path or http(s) URL to the reference audio clip.")
    p.add_argument("--ref-text", help="Transcript of ref_audio (required for --mode icl).")
    p.add_argument("--from-dir", default=None,
                   help="Batch mode: precompute every (name.wav, name.txt) pair in this directory. "
                        "If set, --name/--ref-audio/--ref-text are ignored.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir)

    if args.from_dir:
        ref_dir = pathlib.Path(args.from_dir)
        if not ref_dir.is_dir():
            print(f"ERROR: --from-dir {ref_dir} is not a directory", file=sys.stderr)
            return 2
        n = batch_from_reference_dir(out_dir, ref_dir, args.mode, args.device)
        print(f"done: {n} voice(s) precomputed into {out_dir}")
        return 0

    if not args.name or not args.ref_audio:
        print("ERROR: either --from-dir, or both --name and --ref-audio", file=sys.stderr)
        return 2
    precompute_single(
        name=args.name,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        mode=args.mode,
        output_dir=out_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
