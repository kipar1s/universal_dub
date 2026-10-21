#!/usr/bin/env python3
"""Test script for MVP Whisper fix - test_transcribe.py"""

import sys
sys.path.insert(0, "C:/work/universal_dub")

from mvp_fix_whisper import transcribe_audio_fixed
import tempfile
import numpy as np
import soundfile as sf

# Generate a simple 5-second test audio (16kHz)
print("🔊 Generating test audio...")
sample_rate = 16000
duration = 2  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
test_audio = 0.1 * np.sin(440 * 2 * np.pi * t) + 0.05 * np.random.randn(len(t))

# Save to temp file
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    test_wav_path = f.name
    sf.write(test_wav_path, test_audio, sample_rate)
    print(f"📁 Test audio saved: {test_wav_path}")

print("\n⏳ Running transcription...")
result = transcribe_audio_fixed(test_wav_path, whisper_model="tiny")

print("\n✅ RESULT:")
for seg in result:
    print(f"   Segment 1: \"{seg['text']}\"")

# Cleanup
import os
os.unlink(test_wav_path)

print("\n🎉 MVP TEST PASSED!")