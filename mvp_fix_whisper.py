#!/usr/bin/env python3
"""
MVP Fix for universal_dub — Replace transcribe_audio with Transformers-based Whisper
This fixes the system whisper.py conflict issue.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def transcribe_audio_fixed(audio_path, whisper_model="tiny"):
    """Fixed version: Uses Transformers directly (no system whisper.py needed)."""
    
    log = print  # Simple logging for MVP
    
    model_name_map = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base", 
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium"  # Changed default to tiny for MVP stability
    }
    
    model_name = model_name_map.get(whisper_model, "openai/whisper-tiny")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[INFO] Loading Whisper {model_name} on {device}")
    
    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load audio with soundfile (already imported in universal_dub.py)
    import soundfile as sf
    wav_path = str(audio_path)
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz (Whisper requirement)
    import scipy.signal as signal
    resampler = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
    audio = resampler
    
    # Normalize and pad
    max_length = 300000  # 18.75 minutes at 16kHz
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_features
    
    # Generate transcription
    generated_ids = model.generate(input_features, max_new_tokens=50)
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    log(f"[INFO] Transcription: {transcript}")
    return [{"text": transcript}]

# Save to file for testing with subprocess
if __name__ == "__main__":
    print("Save this function as transcribe_audio in universal_dub.py")