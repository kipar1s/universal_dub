#!/usr/bin/env python3
"""
universal_dub — Universal Video Dubbing Pipeline (any Whisper language -> any TTS language)

Pipeline:
  Video → Audio Extract (ffmpeg)
    → Whisper Transcribe (source language: auto or --source-lang)
      → Diarization (F0 + GMM → male/female per segment)
        → LLM Translation (Hy-MT, source → target language)
          → TTS (Qwen3-TTS local GPU [10 langs] / Edge-TTS cloud [75 langs], per-gender voices)
            → Merge (atempo 0.9–1.2x clamp, 24kHz) + Video → Final dubbed video

Usage:
  python universal_dub.py video.mp4 --source-lang en --target-lang ru
  python universal_dub.py video.mp4 --target-lang de            # source: auto-detect
  python universal_dub.py video.mp4 --target-lang uk --tts-engine edge
"""

import os, sys, json, re, time, subprocess, argparse, shutil, wave
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ─── ffmpeg ──────────────────────────────────────────────────────
_FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"
if not os.path.isfile(_FFMPEG_EXE):          # портативный fallback: ffmpeg из PATH
    _FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"
os.environ["FFMPEG"] = _FFMPEG_EXE
_ffmpeg_dir = os.path.dirname(_FFMPEG_EXE)
if os.path.isfile(_FFMPEG_EXE) and _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + ";" + os.environ.get("PATH", "")


def log(msg, level="INFO"):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}")


# ─── Languages registry ──────────────────────────────────────────

# ISO → English full name (for Hy-MT prompts)
LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kk": "Kazakh", "ko": "Korean", "mr": "Marathi", "ms": "Malay",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sv": "Swedish", "ta": "Tamil", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh": "Chinese",
}

# Qwen3-TTS supported languages (ISO → Qwen language name)
QWEN_LANGUAGES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}

# Qwen3 speakers (0.6B/1.7B-CustomVoice), by native language
QWEN_SPEAKERS = {
    "zh": {"female": "Serena", "male": "Dylan"},
    "en": {"female": "Vivian", "male": "Ryan"},   # no native female EN speaker; Vivian covers it
    "ja": {"female": "Ono_Anna", "male": "Dylan"},
    "ko": {"female": "Sohee", "male": "Dylan"},
    # languages without native Qwen speakers → best general pair
    "ru": {"female": "Vivian", "male": "Uncle_Fu"},
    "de": {"female": "Vivian", "male": "Uncle_Fu"},
    "fr": {"female": "Vivian", "male": "Uncle_Fu"},
    "es": {"female": "Vivian", "male": "Uncle_Fu"},
    "pt": {"female": "Vivian", "male": "Uncle_Fu"},
    "it": {"female": "Vivian", "male": "Uncle_Fu"},
}
QWEN_ALL_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
                     "Ryan", "Aiden", "Ono_Anna", "Sohee"]

# Edge-TTS default voices per language (ISO → (male, female)); ru as template for others
EDGE_VOICES = {
    "ru": ("ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"),
    "en": ("en-GB-RyanNeural", "en-GB-SoniaNeural"),
    "de": ("de-DE-ConradNeural", "de-DE-KatjaNeural"),
    "fr": ("fr-FR-HenriNeural", "fr-FR-DeniseNeural"),
    "es": ("es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
    "pt": ("pt-BR-AntonioNeural", "pt-BR-ThelmaNeural"),
    "it": ("it-IT-DiegoNeural", "it-IT-ElsaNeural"),
    "ja": ("ja-JP-KeitaNeural", "ja-JP-NanamiNeural"),
    "ko": ("ko-KR-InJoonNeural", "ko-KR-SunHiNeural"),
    "zh": ("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
    "uk": ("uk-UA-PavloNeural", "uk-UA-PolinaNeural"),
    "pl": ("pl-PL-MarekNeural", "pl-PL-ZofiaNeural"),
    "tr": ("tr-TR-AhmetNeural", "tr-TR-EmelNeural"),
    "ar": ("ar-SA-HamedNeural", "ar-SA-ZariyahNeural"),
    "hi": ("hi-IN-MadhurNeural", "hi-IN-SwaraNeural"),
    "vi": ("vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"),
    "nl": ("nl-NL-MaartenNeural", "nl-NL-XeniaNeural"),
    "cs": ("cs-CZ-AntoninNeural", "cs-CZ-VlastaNeural"),
    "sv": ("sv-SE-MattiasNeural", "sv-SE-SveaNeural"),
    "el": ("el-GR-NestorasNeural", "el-GR-AlexiaNeural"),
    "fi": ("fi-FI-HarriNeural", "fi-FI-NooraNeural"),
    "da": ("da-DK-JeppeNeural", "da-DK-ChristelNeural"),
    "he": ("he-IL-AvriNeural", "he-IL-HilaNeural"),
    "th": ("th-TH-PremwadeeNeural", "th-TH-PremwadeeNeural"),
    "id": ("id-ID-ArdiNeural", "id-ID-GadisNeural"),
    "ms": ("ms-MY-OsmanNeural", "ms-MY-YasminNeural"),
    "ro": ("ro-RO-VladuNeural", "ro-RO-AlinaNeural"),
    "hu": ("hu-HU-TamasNeural", "hu-HU-TamásNeural"),
    "fa": ("fa-IR-FaridehNeural", "fa-IR-FaridehNeural"),
    "bg": ("bg-BG-BorislavNeural", "bg-BG-KalinaNeural"),
}


def voices_for(target_lang: str, male_override=None, female_override=None):
    """Return (male_voice, female_voice, engine) for target language.

    Engine choice: Qwen if language is in QWEN_LANGUAGES, else Edge-TTS.
    """
    tl = target_lang.lower()
    male = male_override
    female = female_override
    if tl in QWEN_LANGUAGES:
        return male or QWEN_SPEAKERS[tl]["male"], female or QWEN_SPEAKERS[tl]["female"], "qwen"
    edge = EDGE_VOICES.get(tl) or EDGE_VOICES["en"]  # неизвестный язык → англ. голоса, а не пустые
    return male or edge[0], female or edge[1], "edge"


# ─── TTS Engines ─────────────────────────────────────────────────

# Qwen3-TTS model registry: size -> (CustomVoice dir, Base/voice-clone dir)
QWEN_MODELS = {
    "0.6b": {
        "custom": ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3-TTS-0.6B-CustomVoice"),
        "base":   ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-0.6B-Base"),
    },
    "1.7b": {
        "custom": ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-1.7B-CustomVoice"),
        "base":   ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-1.7B-Base"),
    },
}

MIN_REF_SECONDS = 2.0     # min reference clip for voice clone
MAX_REF_SECONDS = 12.0    # max reference clip


def resolve_qwen_model(size="0.6b", kind="custom", explicit=None):
    """Return local dir if downloaded, else HF id for on-demand download."""
    if explicit:
        return explicit
    hf_id, local_name = QWEN_MODELS[size][kind]
    local_dir = Path(__file__).parent / "models" / local_name
    return str(local_dir) if local_dir.is_dir() else hf_id


class QwenTTS:
    """Qwen3-TTS (local, GPU). 10 languages, 9 premium speakers.

    size:  '0.6b' | '1.7b'
    clone: False -> generate_custom_voice (named speakers)
          True  -> generate_voice_clone (ref_audio per speaker, Base model)
    """

    def __init__(self, size="0.6b", kind=None, model_path=None, device="cuda"):
        import torch
        from qwen_tts import Qwen3TTSModel
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kind = kind or ("base" if model_path and "Base" in Path(model_path).name else "custom")
        model_path = model_path or resolve_qwen_model(size, kind)
        log(f"[QwenTTS] Loading {model_path} on {self.device} (size={size}, kind={kind}) ...")
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
        )
        self.clone = kind == "base"
        self.speaker_ref = {}   # {speaker: (audio_path, ref_text)}
        self.prompts = {}       # {speaker: VoiceClonePromptItem} (предвычислены один раз)
        if self.clone:
            log("[QwenTTS] Voice-clone mode (Base model)")
        else:
            log(f"[QwenTTS] Ready. Languages: {self.model.get_supported_languages()}")
            try:
                log(f"[QwenTTS] Speakers: {self.model.get_supported_speakers()}")
            except Exception:
                pass

    def set_references(self, speaker_ref):
        """speaker_ref: {speaker_label: (audio_path, ref_text)} for clone mode."""
        self.speaker_ref = dict(speaker_ref or {})
        log(f"[QwenTTS] Voice refs set for: {list(self.speaker_ref)}")
        self._prepare_prompts()

    def _prepare_prompts(self):
        """Один раз на спикера закодировать референс (VQ-коды + speaker embedding).

        Без этого каждый вызов generate_voice_clone() заново читает и
        энкодирует весь клип референса (12 с аудио) — это и была главная
        причина медленного TTS на 400+ сегментах.
        """
        if not self.clone or not self.speaker_ref:
            return
        ok = 0
        for spk, (audio, text) in self.speaker_ref.items():
            try:
                items = self.model.create_voice_clone_prompt(
                    ref_audio=str(audio), ref_text=text, x_vector_only_mode=False)
                if items:
                    self.prompts[spk] = items[0]
                    ok += 1
            except Exception as e:
                log(f"[QwenTTS] prompt prep failed for {spk}: {e}", "WARN")
        if ok:
            log(f"[QwenTTS] Voice-clone prompts ready: {ok}/{len(self.speaker_ref)}")
        else:
            log("[QwenTTS] no precomputed prompts — encoding per call (slower)", "WARN")

    def _clone_prompt_for(self, voice):
        item = self.prompts.get(voice)
        if item is None and self.prompts:
            item = next(iter(self.prompts.values()))
        return item

    def generate_sync(self, text, output_path, language=None, voice=None):
        import soundfile as sf
        lang = language or "Auto"
        if self.clone:
            item = self._clone_prompt_for(voice)
            if item is not None:
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=lang,
                    voice_clone_prompt=[item], non_streaming_mode=True)
            else:
                # fallback: библиотека сама кодирует референс на лету (медленнее)
                ref = None
                if self.speaker_ref:
                    ref = self.speaker_ref.get(voice) or next(iter(self.speaker_ref.values()))
                if ref is None:
                    raise RuntimeError("voice-clone mode: no reference audio set "
                                       "(set via QwenTTS.set_references)")
                ref_audio, ref_text = ref
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=lang,
                    ref_audio=str(ref_audio), ref_text=ref_text,
                    non_streaming_mode=True)
        else:
            wavs, sr = self.model.generate_custom_voice(
                text=text, language=lang, speaker=voice)
        sf.write(str(output_path), np.asarray(wavs[0], dtype=np.float32), sr)

    def generate_batch_sync(self, items, output_paths, language=None, batch_size=8):
        """Пакетное озвучивание списка (text, voice) — батчами по batch_size.

        Один model.generate() на чанк (padded batch) заметно быстрее N
        последовательных вызовов. Сбой батча → автоматический fallback
        на поштучный generate_sync(). Возвращает (ok, fail).
        """
        if len(items) != len(output_paths):
            raise ValueError("items/output_paths length mismatch")
        import soundfile as sf
        lang = language or "Auto"
        ok = fail = 0
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            paths = output_paths[i:i + batch_size]
            texts = [t for t, _ in chunk]
            try:
                if self.clone:
                    chunk_prompts = [self._clone_prompt_for(v) for _, v in chunk]
                    if any(p is None for p in chunk_prompts):
                        raise RuntimeError("no precomputed voice-clone prompts")
                    wavs, sr = self.model.generate_voice_clone(
                        text=texts, language=lang,
                        voice_clone_prompt=chunk_prompts, non_streaming_mode=True)
                else:
                    wavs, sr = self.model.generate_custom_voice(
                        text=texts, language=lang,
                        speaker=[v for _, v in chunk])
                if len(wavs) != len(texts):
                    raise RuntimeError(f"expected {len(texts)} wavs, got {len(wavs)}")
                for wav, p in zip(wavs, paths):
                    sf.write(str(p), np.asarray(wav, dtype=np.float32), sr)
                ok += len(chunk)
            except Exception as e:
                log(f"[QwenTTS] batch of {len(chunk)} failed ({e}) — per-segment fallback", "WARN")
                for (t, v), p in zip(chunk, paths):
                    try:
                        self.generate_sync(t, str(p), language=language, voice=v)
                        ok += 1
                    except Exception as e2:
                        fail += 1
                        log(f"[QwenTTS] segment failed: {e2}", "WARN")
        return ok, fail


class EdgeTTS:
    """Edge-TTS (cloud). 75 languages, many neural voices per language."""

    def __init__(self, voice="en-US-AriaNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume

    def generate_sync(self, text, output_path, language=None, voice=None):
        import asyncio, edge_tts
        async def _run():
            comm = edge_tts.Communicate(text, voice or self.voice,
                                        rate=self.rate, volume=self.volume)
            await comm.save(str(output_path))
        asyncio.run(_run())


class Sapi5TTS:
    """Windows SAPI5 last resort (1 language)."""

    def generate_sync(self, text, output_path, language=None, voice=None):
        import pyttsx3
        eng = pyttsx3.init()
        try:
            eng.save_to_file(text, output_path)
            eng.runAndWait()
        finally:
            eng.stop()


# ─── Data model ──────────────────────────────────────────────────

@dataclass
class Segment:
    index: int
    start_ms: float
    end_ms: float
    speaker: str = "SPEAKER_00"
    original_text: str = ""
    corrected_text: str = ""
    translated_text: str = ""
    tts_file: Optional[str] = None
    gender: Optional[str] = None      # "male" | "female"
    voice: Optional[str] = None       # per-segment voice name
    f0_hz: Optional[float] = None


# ─── Stage 1: Audio extraction ───────────────────────────────────

def extract_audio(video_path, out_wav, sample_rate=16000):
    cmd = [_FFMPEG_EXE, "-y", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out_wav)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audio extract failed: {r.stderr[-300:]}")
    log(f"Audio extracted: {Path(out_wav).name}")


def get_video_duration(video_path):
    ffprobe = str(Path(_FFMPEG_EXE).parent / "ffprobe.exe")
    exe = ffprobe if os.path.exists(ffprobe) else _FFMPEG_EXE
    r = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) * 1000
    except ValueError:
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
        if m:
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) * 1000
        raise RuntimeError(f"could not determine video duration: {r.stdout!r} {r.stderr[-200:]!r}")


# ─── Stage 2: Whisper transcription ─────────────────────────────

def group_words_into_segments(raw_segments):
    """Group Whisper word-level results into speech segments (gap < 500ms).

    Whisper word objects use key 'word' (not 'text')!
    """
    gap_threshold_ms = 500

    all_words = []
    for seg in raw_segments:
        start_s = seg["start"]
        end_s = seg["end"]
        if "words" in seg:
            for word in seg["words"]:
                all_words.append({
                    "word": word.get("word", word.get("text", "")),
                    "start_ms": word.get("start", start_s) * 1000,
                    "end_ms": word.get("end", end_s) * 1000,
                })
        else:
            text = seg.get("text", "").strip()
            if text:
                all_words.append({"word": text, "start_ms": start_s * 1000,
                                 "end_ms": end_s * 1000})

    if not all_words:
        return []

    segments = []
    current = {"words": [all_words[0]], "start_ms": all_words[0]["start_ms"],
               "end_ms": all_words[0]["end_ms"]}
    for word in all_words[1:]:
        gap = word["start_ms"] - current["end_ms"]
        if gap > gap_threshold_ms and not word["word"].strip().endswith((".", "?", "!")):
            segments.append(current)
            current = {"words": [word], "start_ms": word["start_ms"],
                       "end_ms": word["end_ms"]}
        else:
            current["words"].append(word)
            current["end_ms"] = word["end_ms"]
    segments.append(current)

    result = []
    for idx, seg in enumerate(segments):
        text = "".join(w["word"] for w in seg["words"]).strip()
        if not text:
            continue
        result.append(Segment(index=idx + 1, start_ms=seg["start_ms"],
                              end_ms=seg["end_ms"], speaker="SPEAKER_00",
                              original_text=text, corrected_text=text))
    for i, s in enumerate(result, 1):
        s.index = i
    return result


def transcribe_audio(audio_path, whisper_model="medium", language=None, device_hint="auto"):
    import whisper
    log(f"Loading Whisper model '{whisper_model}'...")
    device = "cuda" if (device_hint in ("auto", "cuda") and
                        __import__("torch").cuda.is_available()) else "cpu"
    model = whisper.load_model(whisper_model, device=device)
    log(f"Transcribing (language={language or 'auto-detect'})...")
    result = model.transcribe(str(audio_path), language=language,
                              task="transcribe", word_timestamps=True,
                              verbose=False, fp16=(device == "cuda"))
    log(f"Detected language: {result.get('language')}")
    segments = group_words_into_segments(result["segments"])
    log(f"Transcription: {len(segments)} segments")
    return segments


# ─── Stage 2.5: Diarization (copied from voice_translate) ───────

def _segment_f0(audio, sr, start_ms, end_ms, pad_s=0.2):
    import librosa
    s = max(0, int(start_ms / 1000.0 * sr) - int(pad_s * sr))
    e = min(len(audio), int(end_ms / 1000.0 * sr) + int(pad_s * sr))
    if e - s < int(0.6 * sr):
        return None
    x = audio[s:e]
    f0, voiced, _ = librosa.pyin(x, fmin=50.0, fmax=500.0, sr=sr, frame_length=2048)
    vals = f0[voiced]
    return float(np.median(vals)) if len(vals) >= 5 else None


def _cluster_f0(f0_by_seg, max_clusters=4):
    from sklearn.mixture import GaussianMixture
    keys = list(f0_by_seg.keys())
    X = np.array([np.log10(v) for v in f0_by_seg.values()]).reshape(-1, 1)
    best = None
    for k in range(1, min(max_clusters, len(keys)) + 1):
        if k == 1:
            label, bic = np.zeros(len(keys), dtype=int), float("inf")
        else:
            g = GaussianMixture(n_components=k, n_init=5, random_state=0).fit(X)
            bic, label = g.bic(X), g.predict(X)
        if best is None or bic < best[0]:
            best = (bic, k, label)
    return {k: int(l) for k, l in zip(keys, best[2])}, best[1]


def assign_speakers(segments, audio_path, male_voice, female_voice, max_clusters=4):
    """Gender + voice per segment. F0 (pyin) + GMM on log-F0."""
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    f0_by_seg = {s.index: _segment_f0(audio, sr, s.start_ms, s.end_ms)
                 for s in segments}
    f0_by_seg = {k: v for k, v in f0_by_seg.items() if v is not None}

    if len(f0_by_seg) >= 6:
        cluster_of, n_clusters = _cluster_f0(f0_by_seg, max_clusters)
    else:
        cluster_of, n_clusters = {k: 0 for k in f0_by_seg}, 1

    clusters = {}
    for idx, c in cluster_of.items():
        clusters.setdefault(c, []).append(f0_by_seg[idx])
    gender_of = {c: ("male" if np.median(v) < 160.0 else "female") for c, v in clusters.items()}

    def _label(idx):
        if idx in cluster_of:
            return gender_of[cluster_of[idx]], cluster_of[idx]
        # nearest voiced neighbour
        segs = {s.index: s for s in segments}
        if segs.get(idx) and f0_by_seg:
            best = min(f0_by_seg.keys(),
                       key=lambda j: abs(segs[j].start_ms - segs[idx].start_ms))
            return gender_of[cluster_of[best]], cluster_of[best]
        return "female", 0

    speaker_counter = {}
    for seg in segments:
        gender, c = _label(seg.index)
        speaker_counter.setdefault(c, len(speaker_counter) + 1)
        seg.speaker = f"SPEAKER_{speaker_counter[c]:02d}"
        seg.gender = gender
        seg.voice = male_voice if gender == "male" else female_voice
        seg.f0_hz = f0_by_seg.get(seg.index)

    summary = {
        "clusters": n_clusters,
        "per_cluster": {f"SPEAKER_{speaker_counter[c]:02d}":
                        {"segments": len(v),
                         "median_f0_hz": round(float(np.median(v)), 1),
                         "gender": gender_of[c]} for c, v in sorted(clusters.items())},
        "voiced_segments": len(f0_by_seg),
        "total_segments": len(segments),
    }
    return segments, summary


# ─── Stage 3: Translation (Hy-MT, parameterized source/target) ──

def translate_segments(segments, hy_mt_model_path, src_lang, tgt_lang,
                       batch_size=3, config=None):
    """Translate segments source → target. src_lang/tgt_lang: ISO codes or None (auto)."""
    from llama_cpp import Llama

    if not os.path.exists(hy_mt_model_path):
        log(f"Hy-MT model not found ({hy_mt_model_path}) — using original text", "WARN")
        for s in segments:
            s.translated_text = s.corrected_text or s.original_text
        return segments

    to_translate = [s for s in segments if s.corrected_text.strip()]
    if not to_translate:
        return segments

    n_ctx = (config or {}).get("translation", {}).get("n_ctx", 16384)
    log(f"Loading Hy-MT (n_ctx={n_ctx})...")

    llm = None
    try:
        import llama_cpp, torch
        gpu_ok = llama_cpp.llama_supports_gpu_offload() and torch.cuda.is_available()
        if gpu_ok:
            log("Hy-MT: GPU offload")
            llm = Llama(model_path=hy_mt_model_path, n_ctx=n_ctx, n_gpu_layers=-1,
                        verbose=False, use_mlock=True)
        else:
            log("Hy-MT: CPU (llama-cpp built without CUDA)")
    except Exception as e:
        log(f"GPU load failed ({e}), CPU fallback", "WARN")

    if llm is None:
        llm = Llama(model_path=hy_mt_model_path, n_ctx=n_ctx, n_gpu_layers=0,
                    verbose=False, use_mlock=True)

    src_name = LANG_NAMES.get((src_lang or "").lower())
    tgt_name = LANG_NAMES.get((tgt_lang or "").lower())
    if src_name and tgt_name:
        prompt_prefix = f"Translate the following {src_name} segment into {tgt_name}, without additional explanation.\n\n"
    else:
        prompt_prefix = f"Translate the following segment into {tgt_name or 'the target language'}, without additional explanation.\n\n"
    log(f"Translation: {src_name or 'auto'} → {tgt_name or 'auto'}")

    # target-script sanity check (Cyrillic/latin/CJK ranges)
    def has_target_script(text):
        if not tgt_lang:
            return True
        tl = tgt_lang.lower()
        if tl in ("ru", "uk", "be", "kk", "mn", "bg"):
            return any('\u0400' <= c <= '\u04FF' for c in text)
        if tl in ("zh", "ja"):
            return any('\u4E00' <= c <= '\u9FFF' for c in text)
        if tl == "ko":
            return any('\uAC00' <= c <= '\uD7A3' for c in text)
        if tl == "ja":
            return any('\u3040' <= c <= '\u30FF' for c in text)
        if tl == "ar":
            return any('\u0600' <= c <= '\u06FF' for c in text)
        if tl == "he":
            return any('\u0590' <= c <= '\u05FF' for c in text)
        if tl == "hi":
            return any('\u0900' <= c <= '\u097F' for c in text)
        return True  # latin-based: accept any non-empty text

    ok = fail = 0

    def _clean(t):
        # strip quotes/markdown artifacts LLMs sometimes add
        t = t.strip()
        if len(t) > 2 and t[0] in "\"'“”‘’" and t[-1] in "\"'“”‘’":
            t = t[1:-1]
        return t.strip()

    def _one(seg):
        """Translate a single segment. Returns (text, ok_flag)."""
        out = llm(prompt=prompt_prefix + seg.corrected_text.strip(),
                  max_tokens=256, temperature=0.3, repeat_penalty=1.1, echo=False)
        result = _clean("".join(c["text"] for c in out["choices"]))
        if result and has_target_script(result):
            return result, True
        return (seg.corrected_text or seg.original_text), False

    def _batch(batch):
        """Translate several segments in ONE prompt. Returns (results, all_ok)."""
        lines = "\n".join(f"{i+1}. {s.corrected_text.strip()}"
                          for i, s in enumerate(batch))
        prompt = (f"Translate the {len(batch)} numbered {src_name or ''} lines into "
                  f"{tgt_name or 'the target language'}.\n"
                  f"Rules: keep the same number of numbered lines, same numbers, "
                  f"only the translation on each line, no explanations.\n\n{lines}")
        out = llm(prompt=prompt,
                  max_tokens=64 * len(batch) + 64, temperature=0.3,
                  repeat_penalty=1.1, echo=False)
        text = _clean("".join(c["text"] for c in out["choices"]))
        # parse numbered lines
        parts = {}
        for line in text.splitlines():
            m = re.match(r"^\s*(\d+)[\.\):\-]?\s+(.+)$", line)
            if m:
                parts[int(m.group(1))] = _clean(m.group(2))
        if len(parts) < len(batch):
            return None  # caller falls back to per-segment
        results = []
        for i, s in enumerate(batch):
            t = parts.get(i + 1, "")
            results.append((t, bool(t and has_target_script(t))))
        return results, True

    # batch sizing: 1 = per-segment, 3 = default (short segments per prompt)
    BATCH = max(1, int(batch_size))
    i = 0
    while i < len(to_translate):
        batch = to_translate[i:i + BATCH]
        done = 0
        try:
            res = _batch(batch)
            if res:
                results, _ = res
                for s, (t, good) in zip(batch, results):
                    s.translated_text = t
                    if good:
                        ok += 1
                    else:
                        fail += 1
                    done += 1
        except Exception as e:
            log(f"batch failed at seg {batch[0].index}: {e}", "WARN")
        if done == 0:  # fallback: per-segment
            for s in batch:
                try:
                    t, good = _one(s)
                    s.translated_text = t
                    ok += good
                    fail += (not good)
                except Exception as e:
                    s.translated_text = s.corrected_text or s.original_text
                    fail += 1
                    log(f"seg {s.index} failed: {e}", "WARN")
                done += 1
        i += len(batch)
        if i % 15 < len(batch) or i >= len(to_translate):
            log(f"Translation: {i}/{len(to_translate)} ({ok} OK, {fail} fallback)")

    for seg in segments:
        if not seg.translated_text.strip():
            seg.translated_text = seg.corrected_text or seg.original_text
    return segments


# ─── Stage 4: TTS generation ─────────────────────────────────────

def select_voice_references(segments, audio_wav, artifacts_dir):
    """Pick the best reference clip per speaker for voice cloning.

    Strategy: per speaker, choose the longest voiced segment with
    2s <= duration <= 12s (fallback: longest >= 2s, trimmed to 12s).
    Clips are cut from the ORIGINAL audio with ffmpeg.

    Returns {speaker_label: (clip_path, ref_text)} or {} if no usable refs.
    """
    from collections import defaultdict
    by_spk = defaultdict(list)
    for s in segments:
        if s.original_text.strip():
            by_spk[s.speaker].append(s)

    refs = {}
    clip_dir = Path(artifacts_dir) / "voice_refs"
    clip_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _FFMPEG_EXE if os.path.isfile(_FFMPEG_EXE) else "ffmpeg"

    def _usable(s):
        d = (s.end_ms - s.start_ms) / 1000.0
        alpha = sum(1 for c in s.original_text if c.isalpha())
        return d, alpha

    for spk, segs in by_spk.items():
        segs.sort(key=lambda s: s.end_ms - s.start_ms, reverse=True)
        # prefer: 2-12s duration AND substantive text (>= 12 alpha chars);
        # else any 2-12s clip; else longest >= 2s
        chosen = None
        for s in segs:
            d, a = _usable(s)
            if MIN_REF_SECONDS <= d <= MAX_REF_SECONDS and a >= 12:
                chosen = s
                break
        if chosen is None:
            for s in segs:
                d, _ = _usable(s)
                if MIN_REF_SECONDS <= d <= MAX_REF_SECONDS:
                    chosen = s
                    break
        if chosen is None:
            for s in segs:
                d, _ = _usable(s)
                if d >= MIN_REF_SECONDS:
                    chosen = s
                    break
        if chosen is None:
            continue
        start = max(0.0, chosen.start_ms / 1000.0 - 0.3)
        dur = min(MAX_REF_SECONDS, max(MIN_REF_SECONDS,
                 (chosen.end_ms - chosen.start_ms) / 1000.0 + 0.4))
        clip_path = clip_dir / f"{spk}_ref.wav"
        cmd = [ffmpeg, "-y", "-i", str(audio_wav), "-ss", f"{start:.2f}",
               "-t", f"{dur:.2f}", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", str(clip_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not Path(clip_path).exists():
            log(f"voice ref clip failed for {spk}: {r.stderr[-200:]}", "WARN")
            continue
        # ref_text ОБЯЗАН быть транскриптом именно этого клипа: Qwen3-TTS в ICL-режиме
        # использует пару (ref_text, ref_audio) вместе. Текст из другого сегмента
        # рассинхронизирован с аудиой и ухудшает клонирование. (Выбор выше уже
        # предпочитает клипы с содержательным текстом >= 12 символов.)
        ref_text = chosen.original_text.strip()
        if sum(1 for c in ref_text if c.isalpha()) < 12:
            log(f"voice ref {spk}: короткий транскрипт ({ref_text!r}) — клон может быть слабее", "WARN")
        refs[spk] = (str(clip_path), ref_text)
        log(f"Voice ref {spk}: {dur:.1f}s clip, text='{ref_text[:60]}'")
    return refs


def create_tts_engine(engine_name, qwen_size="0.6b", qwen_model_path=None,
                      voice_clone=False, edge_voice="en-US-AriaNeural"):
    if engine_name == "qwen":
        kind = "base" if voice_clone else "custom"
        return QwenTTS(size=qwen_size, kind=kind, model_path=qwen_model_path)
    if engine_name == "edge":
        return EdgeTTS(voice=edge_voice)
    if engine_name == "pyttsx3":
        return Sapi5TTS()
    raise ValueError(f"unknown engine {engine_name}")


def _tts_name(seg):
    return f"{seg.index:04d}_{int(seg.start_ms):06d}-{int(seg.end_ms):06d}.wav"


def _tts_duration_bad(pth, seg):
    """Защита от «разбегавшейся» генерации (batch-режим qwen_tts иногда
    не останавливается на EOS и выдаёт десятки минут тишости/повторов).
    Короткая фраза не может звучать втрое дольше оригинала и дольше 15 с.
    """
    try:
        with wave.open(str(pth), "rb") as wf:
            dur = wf.getnframes() / max(wf.getframerate(), 1)
    except Exception:
        return False
    src = max((seg.end_ms - seg.start_ms) / 1000.0, 1.0)
    return dur > 15.0 and dur > 3.0 * src


def generate_all_tts(segments, output_dir, engine, engine_name, target_lang,
                     tts_batch=1):
    tts_dir = output_dir / "tts_segments"
    tts_dir.mkdir(parents=True, exist_ok=True)
    to_gen = [s for s in segments if s.translated_text.strip()]
    log(f"Generating TTS for {len(to_gen)} segments ({engine_name})...")
    # Qwen wants full language names ('german'), Edge wants ISO
    if engine_name == "qwen" and target_lang:
        tts_language = QWEN_LANGUAGES.get(target_lang.lower(), "Auto")
    else:
        tts_language = target_lang  # ISO or None

    # Resume: сегменты с уже готовым непустым (и нормальным по длительности) wav пропускаем
    pending = []
    for s in to_gen:
        p = Path(s.tts_file) if s.tts_file else tts_dir / _tts_name(s)
        if p.exists() and p.stat().st_size > 200 and not _tts_duration_bad(p, s):
            s.tts_file = str(p)
        else:
            pending.append(s)
    skipped = len(to_gen) - len(pending)
    if skipped:
        log(f"TTS resume: {skipped}/{len(to_gen)} already done, generating {len(pending)}")
    ok = fail = 0
    t0 = time.time()

    batch_fn = getattr(engine, "generate_batch_sync", None)
    if batch_fn and pending and tts_batch > 1:
        # QwenTTS: пакетный вызов (экспериментально — см. --tts-batch)
        items = [(s.translated_text, s.voice) for s in pending]
        paths = [str(tts_dir / _tts_name(s)) for s in pending]
        log(f"TTS batch mode: {len(pending)} segments, batch={tts_batch}")
        batch_fn(items, paths, language=tts_language, batch_size=max(2, tts_batch))
    # Финальный прогон по каждому сегменту: валидация файла (размер + длительность),
    # догенерация по одному сегменту при битом/разбежавшемся wav.
    ok = fail = 0
    t0 = time.time()
    for i, seg in enumerate(pending):
        out_path = tts_dir / _tts_name(seg)
        p = Path(out_path)
        good = p.exists() and p.stat().st_size > 200 and not _tts_duration_bad(p, seg)
        if not good and p.exists():
            log(f"seg {seg.index}: suspicious wav ({p.stat().st_size} B) — regenerating", "WARN")
            p.unlink(missing_ok=True)
        if not good:
            try:
                engine.generate_sync(seg.translated_text, str(out_path),
                                     language=tts_language, voice=seg.voice)
                p = Path(out_path)
                good = p.exists() and p.stat().st_size > 200 and not _tts_duration_bad(p, seg)
            except Exception as e:
                log(f"TTS failed seg {seg.index}: {e}", "WARN")
        if good:
            seg.tts_file = str(out_path)
            ok += 1
        else:
            fail += 1
            log(f"TTS failed seg {seg.index} (no valid audio)", "WARN")
        if (i + 1) % 10 == 0 or i == len(pending) - 1:
            log(f"TTS: {i+1}/{len(pending)} ({ok} OK, {fail} failed) [{time.time()-t0:.0f}s]")
    done_total = ok + skipped
    log(f"TTS complete: {done_total}/{len(to_gen)} in {time.time()-t0:.0f}s")
    if to_gen and done_total == 0:
        raise RuntimeError(
            f"TTS failed for all {len(to_gen)} segments — check engine/voice/language. "
            f"(engine={engine_name}, language={tts_language}, voices={set(s.voice for s in to_gen)})")


# ─── Stage 5: Merge (atempo clamp 0.9–1.2x) ─────────────────────

def merge_tts_segments(segments, video_duration_ms, output_wav):
    log(f"Merging {len(segments)} segments...")
    sr = 24000
    total_samples = int((video_duration_ms / 1000.0) * sr)
    ffmpeg = _FFMPEG_EXE if os.path.isfile(_FFMPEG_EXE) else "ffmpeg"
    temp_dir = Path(output_wav).parent / "temp_tts_stretch"
    temp_dir.mkdir(parents=True, exist_ok=True)

    stretched = []
    for seg in segments:
        if not seg.tts_file or not Path(seg.tts_file).exists():
            continue
        orig = (seg.end_ms - seg.start_ms) / 1000.0
        if orig <= 0.05:
            continue
        try:
            with wave.open(seg.tts_file, "rb") as wf:
                tts_dur = wf.getnframes() / wf.getframerate()
        except Exception:
            tts_dur = 1.0
        stretched_path = temp_dir / f"{seg.index:04d}_stretched.wav"
        MIN_SPEED, MAX_SPEED = 0.9, 1.2
        if abs(tts_dur - orig) > 0.1 and tts_dur > 0.1:
            speed = min(max(tts_dur / orig, MIN_SPEED), MAX_SPEED)
            filters, r = [], speed
            while abs(r - 1.0) > 0.01:
                if r > 2.0:
                    filters.append("atempo=2.0"); r /= 2.0
                elif r < 0.5:
                    filters.append("atempo=0.5"); r /= 0.5
                else:
                    filters.append(f"atempo={min(max(r, 0.5), 2.0):.4f}"); r = 1.0
            cmd = [ffmpeg, "-y", "-i", seg.tts_file, "-af", ",".join(filters),
                   "-ar", str(sr), str(stretched_path)]
            try:
                subprocess.run(cmd, capture_output=True, timeout=120, text=True)
            except subprocess.TimeoutExpired:
                shutil.copy(seg.tts_file, stretched_path)
        else:
            cmd = [ffmpeg, "-y", "-i", seg.tts_file, "-ar", str(sr), str(stretched_path)]
            subprocess.run(cmd, capture_output=True, text=True)
        stretched.append((seg, str(stretched_path)))
    log(f"Stretched {len(stretched)} segments")

    full_audio = np.zeros(total_samples, dtype=np.float32)
    prev_end = 0
    for seg, sp in stretched:
        with wave.open(sp, "rb") as wf:
            data = wf.readframes(wf.getnframes())
            seg_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        start_sample = max(int(seg.start_ms / 1000.0 * sr), prev_end)
        avail = total_samples - start_sample
        n = min(len(seg_audio), avail)
        if n > 0:
            full_audio[start_sample:start_sample + n] += seg_audio[:n].clip(-1, 1)
            prev_end = start_sample + n

    full_audio = np.clip(full_audio, -1.0, 1.0)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((full_audio * 32767).astype(np.int16).tobytes())
    log(f"Merged audio: {Path(output_wav).name}")


# ─── Stage 6: Video + audio ──────────────────────────────────────

def merge_video_audio(video_path, dubbed_wav, output_mp4,
                      remove_original=True, original_vol="0%"):
    if remove_original:
        cmd = [_FFMPEG_EXE, "-y", "-i", str(video_path), "-i", str(dubbed_wav),
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-shortest", str(output_mp4)]
    else:
        cmd = [_FFMPEG_EXE, "-y", "-i", str(video_path), "-i", str(dubbed_wav),
               "-filter_complex",
               f"[0:a]volume={original_vol}[orig];[1:a][orig]amix=inputs=2:duration=shortest[out]",
               "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               str(output_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"video merge failed: {r.stderr[-400:]}")
    log(f"Final video: {Path(output_mp4).name} ({Path(output_mp4).stat().st_size / 1048576:.1f} MB)")


# ─── Pipeline ────────────────────────────────────────────────────

def run_pipeline(video_path, source_lang, target_lang, output_dir,
                 tts_engine="auto", qwen_size="0.6b", qwen_model_path=None,
                 voice_clone=False,
                 male_voice=None, female_voice=None, no_diarization=False,
                 whisper_model="medium", hy_mt_model=None,
                 batch_size=3, tts_batch=1, remove_original=True):
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = video_path.stem

    # voices + engine resolution
    v_male, v_female, engine_hint = voices_for(target_lang or "ru",
                                               male_voice, female_voice)
    engine_name = tts_engine if tts_engine != "auto" else engine_hint
    if engine_name == "qwen" and not (target_lang or "").lower() in QWEN_LANGUAGES:
        log(f"Qwen3-TTS does not speak '{target_lang}' — falling back to Edge-TTS", "WARN")
        engine_name = "edge"
    if engine_name == "edge":
        edge_pair = EDGE_VOICES.get((target_lang or "ru").lower(),
                                    ("en-US-GuyNeural", "en-US-AriaNeural"))
        v_male = male_voice or edge_pair[0]
        v_female = female_voice or edge_pair[1]
    log(f"Engine: {engine_name} | male={v_male} female={v_female} | target={target_lang}")

    log("=" * 70)
    log(f"SOURCE: {video_path.name}")
    log(f"LANGUAGE: {source_lang or 'auto'} → {target_lang}")
    log("=" * 70)

    # Stage 1
    audio_wav = artifacts_dir / f"{ts}_{stem}_audio.wav"
    extract_audio(str(video_path), str(audio_wav))
    video_duration_ms = get_video_duration(str(video_path))

    # Stage 2
    segments = transcribe_audio(str(audio_wav), whisper_model=whisper_model,
                                language=source_lang)
    with open(artifacts_dir / f"{ts}_{stem}_transcript.json", "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)

    # Stage 2.5
    if not no_diarization and segments:
        log("Diarization (F0 + GMM)...")
        try:
            segments, diar = assign_speakers(segments, str(audio_wav),
                                             v_male, v_female)
            log(f"Diarization: {diar['voiced_segments']}/{diar['total_segments']} voiced, "
                f"{diar['clusters']} cluster(s)")
            for spk, info in diar["per_cluster"].items():
                log(f"  {spk}: {info['segments']} segs, F0={info['median_f0_hz']} Hz, {info['gender']}")
        except Exception as e:
            log(f"Diarization failed ({e}) — single voice", "WARN")
            for s in segments:
                s.voice = v_female
                s.gender = "female"
    else:
        for s in segments:
            s.voice = v_female

    # Stage 3
    segments = translate_segments(segments, hy_mt_model, source_lang, target_lang,
                                  batch_size=batch_size)
    with open(artifacts_dir / f"{ts}_{stem}_translated.json", "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=2)

    # Stage 4
    clone_mode = (voice_clone and engine_name == "qwen")
    if clone_mode:
        # voice-clone: key voices by speaker label (SPEAKER_01, ...) not by gender
        for s in segments:
            s.voice = s.speaker
        log("Voice-clone: selecting reference clips per speaker...")
        refs = select_voice_references(segments, str(audio_wav), artifacts_dir)
        if not refs:
            log("No usable voice refs — falling back to named speakers", "WARN")
            clone_mode = False
            for s in segments:
                s.voice = v_female
    edge_voice = v_female
    engine = create_tts_engine(engine_name, qwen_size=qwen_size,
                               qwen_model_path=qwen_model_path,
                               voice_clone=clone_mode, edge_voice=edge_voice)
    if clone_mode:
        engine.set_references(refs)
    generate_all_tts(segments, artifacts_dir, engine, engine_name, target_lang,
                     tts_batch=tts_batch)

    # Stage 5
    dubbed_wav = artifacts_dir / f"{ts}_{stem}_dubbed.wav"
    merge_tts_segments(segments, video_duration_ms, str(dubbed_wav))

    # Stage 6
    out_mp4 = output_dir / f"{stem}_{(target_lang or 'xx').lower()}_dubbed.mp4"
    merge_video_audio(str(video_path), str(dubbed_wav), str(out_mp4), remove_original)

    log("=" * 70)
    log(f"✓ COMPLETE: {out_mp4}")
    log(f"  {len(segments)} segments | {source_lang or 'auto'} → {target_lang} | {engine_name}")
    log("=" * 70)
    return str(out_mp4)


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Universal video dubbing: any Whisper lang -> any TTS lang")
    p.add_argument("video", help="input video file")
    p.add_argument("--source-lang", default=None,
                   help="source language ISO code (en, uk, de, ...). Default: auto-detect")
    p.add_argument("--target-lang", default="ru",
                   help="target language ISO code. Qwen3-TTS: zh,en,ja,ko,de,fr,ru,pt,es,it. "
                        "Edge-TTS fallback: uk,pl,tr,ar,hi,vi,nl,cs,sv,el,fi,da,he,th,id,ms,ro,hu,fa,bg")
    p.add_argument("--tts-engine", default="auto", choices=["auto", "qwen", "edge", "pyttsx3"],
                   help="auto = Qwen if target supported, else Edge (default)")
    p.add_argument("--qwen-size", default="0.6b", choices=["0.6b", "1.7b"],
                   help="Qwen3-TTS model size (default 0.6b; 1.7b = better quality, slower)")
    p.add_argument("--voice-clone", action="store_true",
                   help="clone the original speakers' voices (Qwen3-TTS Base model, needs ~3-10s per speaker)")
    p.add_argument("--qwen-model", default=None,
                   help="explicit Qwen3-TTS model id or local dir (overrides --qwen-size/--voice-clone)")
    p.add_argument("--male-voice", default=None, help="override male voice name")
    p.add_argument("--female-voice", default=None, help="override female voice name")
    p.add_argument("--no-diarization", action="store_true",
                   help="one voice for all segments (default female)")
    p.add_argument("--whisper-model", default="medium",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--hy-mt-model",
                   default=r"G:\work\hermes\vvp-translate\.models\HY-MT1.5-1.8B-Q4_K_M.gguf",
                   help="path to Hy-MT GGUF")
    p.add_argument("--batch-size", type=int, default=3,
                   help="segments per translation prompt (default 3; 1 = one by one, slow but safest)")
    p.add_argument("--tts-batch", type=int, default=1,
                   help="TTS segments per batched Qwen call. 1 = sequential (default, надёжно); "
                        ">1 = experimental batch (qwen_tts 0.1.1 may produce runaway audio in batches, "
                        "suspicious wavs are auto-detected and regenerated one-by-one)")
    p.add_argument("--keep-original-audio", action="store_true",
                   help="mix original under dubbed instead of replacing")
    p.add_argument("--output", default="./output")
    a = p.parse_args()

    run_pipeline(
        video_path=a.video,
        source_lang=(a.source_lang or "").lower() or None,
        target_lang=(a.target_lang or "").lower() or None,
        output_dir=a.output,
        tts_engine=a.tts_engine,
        qwen_size=a.qwen_size,
        qwen_model_path=a.qwen_model,
        voice_clone=a.voice_clone,
        male_voice=a.male_voice,
        female_voice=a.female_voice,
        no_diarization=a.no_diarization,
        whisper_model=a.whisper_model,
        hy_mt_model=a.hy_mt_model,
        batch_size=a.batch_size,
        tts_batch=a.tts_batch,
        remove_original=not a.keep_original_audio,
    )


if __name__ == "__main__":
    main()
