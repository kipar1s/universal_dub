#!/usr/bin/env python3
"""Smoke test: Chatterbox Multilingual V3 — load, generate RU/EN, with & without ref clip."""
import time, sys
from pathlib import Path
import torch

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] Loading ChatterboxMultilingualTTS (v3) on cuda ...", flush=True)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
print(f"[{time.time()-t0:.1f}s] Model loaded. sr={model.sr}", flush=True)

import torchaudio as ta
import numpy as np
import soundfile as sf
out_dir = Path(__file__).parent / "test_out"
out_dir.mkdir(exist_ok=True)

def _save(wav, path):
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    arr = np.asarray(wav, dtype=np.float32)
    sf.write(str(path), np.squeeze(arr), model.sr)  # squeeze: soundfile 0.14 chokes on (1,N)

ref = r"G:\universal_dub\test_out/ref_clip.wav"

tests = [
    ("ru_default", "Привет! Это тест русского языка в системе озвучки.", None, "ru"),
    ("en_default", "Hello! This is a test of the English language in the dubbing system.", None, "en"),
    ("ru_clone", "Привет! Это тест клонирования голоса на русском языке.", ref, "ru"),
]

for name, text, audio_ref, lang in tests:
    t1 = time.time()
    kwargs = dict(text=text, language_id=lang)
    if audio_ref:
        kwargs["audio_prompt_path"] = audio_ref
    wav = model.generate(**kwargs)
    arr = wav.cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav)
    dur = len(arr[0]) / model.sr if arr.ndim > 1 else len(arr) / model.sr
    _save(wav, out_dir / f"{name}.wav")
    print(f"[{time.time()-t0:.1f}s] {name}: generated in {time.time()-t1:.2f}s, dur={dur:.2f}s -> {out_dir/f'{name}.wav'}", flush=True)

print("DONE", flush=True)
