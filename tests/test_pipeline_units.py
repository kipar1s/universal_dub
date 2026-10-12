"""Unit tests for universal_dub pipeline helpers (no GPU / no models needed).

Covers:
- transcribe_audio guard against the conflicting single-file `whisper` package
- _find_artifact (stable names + legacy {ts}_ prefixed names)
- voices_for (engine + voice resolution)
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import universal_dub as ud  # noqa: E402


def test_transcribe_rejects_wrong_whisper_package(tmp_path, monkeypatch):
    """`import whisper` must resolve to openai-whisper (has load_model).

    If the graphite single-file package is installed instead, transcribe_audio
    must fail FAST with an actionable message — not silently return a
    model tuple (the bug that broke every Web UI run on 2026-09-14).
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = MagicMock(is_available=MagicMock(return_value=False))
    fake_whisper = types.ModuleType("whisper")  # no load_model attribute!
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr(ud, "log", lambda *a, **k: None)

    import pytest
    with pytest.raises(RuntimeError, match="not openai-whisper"):
        ud.transcribe_audio(str(tmp_path / "a.wav"))


def test_find_artifact_stable_name(tmp_path):
    (tmp_path / "vid_transcript.json").write_text("[]", encoding="utf-8")
    p = ud._find_artifact(tmp_path, "vid", "_transcript.json")
    assert p is not None and p.name == "vid_transcript.json"


def test_find_artifact_legacy_ts_prefixed(tmp_path):
    (tmp_path / "20260829_174430_vid_transcript.json").write_text("[]", encoding="utf-8")
    p = ud._find_artifact(tmp_path, "vid", "_transcript.json")
    assert p is not None and "20260829" in p.name


def test_find_artifact_missing(tmp_path):
    assert ud._find_artifact(tmp_path, "nope", "_transcript.json") is None


def test_voices_for_qwen_lang():
    m, f, engine = ud.voices_for("de")
    assert engine == "qwen"
    assert m == "Uncle_Fu" and f == "Vivian"


def test_voices_for_edge_fallback_lang():
    m, f, engine = ud.voices_for("uk")
    assert engine == "edge"
    assert m == "uk-UA-PavloNeural" and f == "uk-UA-PolinaNeural"


def test_voices_for_unknown_lang_falls_back_to_english():
    m, f, engine = ud.voices_for("xx")
    assert engine == "edge"
    assert m and f  # non-empty English voices, not None


def test_voices_for_override():
    m, f, engine = ud.voices_for("ru", male_override="MyVoice")
    assert m == "MyVoice" and f == "Vivian"
