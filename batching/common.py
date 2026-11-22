"""Shared helpers: load the real model (or a scaled-down copy for testing) and audio."""
import os

import numpy as np
import torch

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


_CACHE: dict = {}


def load_model(small: bool = False, device: str | None = None):
    """Return (model, feature_extractor, tokenizer_or_None), loading each model only once.

    Repeated calls in the same process reuse the loaded model instead of loading a
    second copy (two copies of the 600M model can exhaust a laptop's memory).
    The model is read-only during inference, so sharing it is safe.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (small, device)
    if key not in _CACHE:
        _CACHE[key] = _load_model(small, device)
    return _CACHE[key]


def _load_model(small: bool, device: str):
    """Return (model, feature_extractor, tokenizer_or_None).

    small=True builds the SAME architecture scaled down with random weights
    (no download). The streaming/caching/decoding math is identical, so it's
    valid for exact-match tests, but its text output is meaningless.
    """
    from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
        NemotronAsrStreamingFeatureExtractor,
    )
    if small:
        from transformers import NemotronAsrStreamingConfig, NemotronAsrStreamingForRNNT
        torch.manual_seed(0)
        vocab = 65
        cfg = NemotronAsrStreamingConfig(
            encoder_config=dict(hidden_size=128, num_hidden_layers=3, num_attention_heads=4,
                                num_key_value_heads=4, intermediate_size=256,
                                subsampling_conv_channels=32, dropout=0.0, layerdrop=0.0,
                                activation_dropout=0.0, attention_dropout=0.0),
            decoder_hidden_size=64, num_decoder_layers=2, vocab_size=vocab, blank_token_id=vocab - 1,
            pad_token_id=vocab - 1)
        model = NemotronAsrStreamingForRNNT(cfg).eval().to(device)
        # The released model's generation_config starts decoding from <blank>; mirror that.
        model.generation_config.decoder_start_token_id = vocab - 1
        model.generation_config.pad_token_id = vocab - 1
        # Calibrate the random output layer so it behaves like real speech: mostly <blank>
        # (move to the next frame) with ~0.3 tokens per frame, so BOTH decoding paths are tested.
        with torch.no_grad():
            torch.manual_seed(1)
            model.joint.head.weight.normal_(0, 0.5)
            model.joint.head.bias.zero_()
            model.joint.head.bias[vocab - 1] = 3.0
            # Trained models' decoders react strongly to the token just written; random tiny
            # embeddings don't. Give them realistic spread so "one token, then move on" happens too.
            torch.manual_seed(3)
            model.decoder.embedding.weight.normal_(0, 1.0)
            model.decoder.decoder_projector.weight.normal_(0, 0.6)
        return model, NemotronAsrStreamingFeatureExtractor(), None

    from transformers import AutoModelForRNNT, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID).eval().to(device)
    return model, processor.feature_extractor, processor.tokenizer


def expand_audio_paths(args: list[str]) -> list[str]:
    """Accept files, folders and wildcards on every OS.

    Windows PowerShell passes "../samples/*.wav" through literally, so we expand
    it here. A folder means "every .wav inside it".
    """
    import glob
    found = []
    for a in args:
        if os.path.isdir(a):
            found += sorted(glob.glob(os.path.join(a, "*.wav")))
        elif any(ch in a for ch in "*?["):
            found += sorted(glob.glob(a))
        else:
            found.append(a)
    missing = [f for f in found if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"Audio file not found: {missing[0]}")
    if args and not found:
        raise SystemExit(f"No .wav files matched {args}. Check the folder path "
                         f"(current folder: {os.getcwd()})")
    return found


def load_audio_or_noise(path=None, seed=0, seconds=6.0):
    """Load a 16 kHz mono file, or synthesize speech-like test audio (no file needed)."""
    if path:
        import soundfile as sf
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // g, rate // g).astype(np.float32)
        return audio
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    t = np.arange(n) / 16000
    # Bursts of harmonic tones with varying pitch + noise: varied enough that tokens differ between clips.
    pitch = 120 + 80 * np.sin(2 * np.pi * rng.uniform(0.2, 0.7) * t)
    env = (np.sin(2 * np.pi * rng.uniform(1.5, 3.5) * t) > -0.2).astype(np.float32)
    phase = 2 * np.pi * np.cumsum(pitch) / 16000
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.2 * env * voice + 0.02 * rng.standard_normal(n)).astype(np.float32)
