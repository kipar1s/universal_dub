"""Regression tests for the 2026-10-15 review fixes.

Covers:
- _tts_duration_bad: OR-logic (runaway >60s OR >3x src AND >5s) — the old
  AND-logic missed runaways in the 15-60s band
- merge_video_audio: keep-original path must mix at an audible volume
  (was original_vol="0%" — the checkbox did nothing)
- EdgeTTS.generate_sync: must produce a real WAV (edge_tts writes MP3;
  pipeline readers use stdlib wave)
- EDGE_VOICES data integrity: male != female, no duplicate genders
"""
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import universal_dub as ud


def _mk_wav(path, seconds, sr=24000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(seconds * sr))


def _seg(start_ms=0, end_ms=2000):
    return ud.Segment(index=1, start_ms=start_ms, end_ms=end_ms,
                      original_text="hi", corrected_text="hi")


# ─── _tts_duration_bad ───────────────────────────────────────────

def test_duration_bad_runaway_over_60s(tmp_path):
    p = tmp_path / "a.wav"
    _mk_wav(p, 70)
    assert ud._tts_duration_bad(p, _seg(0, 2000)) is True


def test_duration_bad_mid_band_runaway(tmp_path):
    """20 s audio for a 2 s segment: old AND-logic (dur>15 AND >3x) caught it,
    new logic must too (20 > 3*2 and 20 > 5)."""
    p = tmp_path / "a.wav"
    _mk_wav(p, 20)
    assert ud._tts_duration_bad(p, _seg(0, 2000)) is True


def test_duration_ok_legitimate_long_translation(tmp_path):
    """8 s audio for a 5 s segment (1.6x) — legitimate, must NOT be flagged."""
    p = tmp_path / "a.wav"
    _mk_wav(p, 8)
    assert ud._tts_duration_bad(p, _seg(0, 5000)) is False


def test_duration_ok_short_segment_slightly_long(tmp_path):
    """4 s audio for a 1 s segment: >3x but under the 5 s absolute floor —
    short segments legitimately stretch; must NOT be flagged."""
    p = tmp_path / "a.wav"
    _mk_wav(p, 4)
    assert ud._tts_duration_bad(p, _seg(0, 1000)) is False


def test_duration_bad_unreadable_is_not_flagged(tmp_path):
    p = tmp_path / "broken.wav"
    p.write_bytes(b"not a wav")
    assert ud._tts_duration_bad(p, _seg()) is False


# ─── merge_video_audio keep-original ─────────────────────────────

def test_keep_original_mixes_at_audible_volume(monkeypatch):
    import inspect
    sig = inspect.signature(ud.merge_video_audio)
    vol = sig.parameters["original_vol"].default
    assert vol != "0%", "keep-original would mix the source at 0% (silent)"
    assert vol.endswith("%") and int(vol.rstrip("%")) > 0


def test_merge_video_audio_passes_volume_to_ffmpeg(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        r = MagicMock()
        r.returncode = 0
        return r

    monkeypatch.setattr(ud.subprocess, "run", fake_run)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")  # stat() must work for the log line
    ud.merge_video_audio("v.mp4", "d.wav", str(out), remove_original=False)
    joined = " ".join(calls[0])
    assert "volume=35%" in joined


# ─── EdgeTTS mp3 -> wav ──────────────────────────────────────────

def test_edge_tts_converts_mp3_to_wav(tmp_path, monkeypatch):
    """edge_tts.save always writes MP3; generate_sync must convert to a real
    WAV via ffmpeg so stdlib wave readers (duration check, merge) work."""
    fake_comm = MagicMock()

    async def fake_save(path):
        Path(path).write_bytes(b"ID3fake-mp3")  # simulates MP3 output

    fake_comm.return_value.save = fake_save
    import edge_tts
    monkeypatch.setattr(edge_tts, "Communicate", fake_comm)

    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        # emulate ffmpeg producing the wav
        _mk_wav(cmd[-1], 1)
        r = MagicMock()
        r.returncode = 0
        r.stderr = ""
        return r

    monkeypatch.setattr(ud.subprocess, "run", fake_run)

    out = tmp_path / "seg.wav"
    eng = ud.EdgeTTS(voice="ru-RU-DmitryNeural")
    eng.generate_sync("привет", str(out))

    assert captured["cmd"][0] == ud._FFMPEG_EXE
    assert "pcm_s16le" in captured["cmd"]
    with wave.open(str(out), "rb") as wf:  # must be a real WAV now
        assert wf.getnframes() > 0
    # temp mp3 cleaned up
    assert not (tmp_path / "seg.wav.mp3.tmp").exists()


def test_edge_tts_conversion_failure_raises(tmp_path, monkeypatch):
    fake_comm = MagicMock()

    async def fake_save(path):
        Path(path).write_bytes(b"ID3fake-mp3")

    fake_comm.return_value.save = fake_save
    import edge_tts
    monkeypatch.setattr(edge_tts, "Communicate", fake_comm)

    def fake_run(cmd, **kw):
        r = MagicMock()
        r.returncode = 1
        r.stderr = "boom"
        return r

    monkeypatch.setattr(ud.subprocess, "run", fake_run)
    eng = ud.EdgeTTS(voice="ru-RU-DmitryNeural")
    with pytest.raises(RuntimeError, match="mp3->wav"):
        eng.generate_sync("привет", str(tmp_path / "seg.wav"))


# ─── EDGE_VOICES data integrity ──────────────────────────────────

def test_edge_voices_male_female_distinct():
    """th/fa historically had the SAME female voice for both genders,
    silently killing gender diarization on Edge."""
    for lang, (m, f) in ud.EDGE_VOICES.items():
        assert m != f, f"{lang}: male and female Edge voices are identical"


def test_edge_voices_names_wellformed():
    import re
    for lang, (m, f) in ud.EDGE_VOICES.items():
        for v in (m, f):
            assert re.fullmatch(r"[a-z]{2}-[A-Z]{2}-[A-Za-z]+Neural", v), \
                f"{lang}: malformed voice name {v!r}"
