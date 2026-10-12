"""Unit tests for universal_dub batch translation (mocked LLM)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import universal_dub as ud


def _make_segments(n=5):
    segs = []
    for i in range(1, n + 1):
        s = ud.Segment(index=i, start_ms=i * 1000, end_ms=(i + 1) * 1000,
                       original_text=f"English sentence number {i}.",
                       corrected_text=f"English sentence number {i}.")
        segs.append(s)
    return segs


class MockLlama:
    """Simulates Hy-MT: batch prompt -> numbered German lines; single -> one line."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, prompt, **kw):
        import re
        if "numbered" in prompt:
            n = int(re.search(r"Translate the (\d+) numbered", prompt).group(1))
            lines = [f"{i+1}. Der deutsche Satz Nummer {i+1}." for i in range(n)]
            text = "\n".join(lines)
        else:
            m = re.search(r"\n\n(.+)$", prompt.strip())
            text = "Der deutsche Satz."
            if m:
                n = m.group(1).split()[-1]
                text = f"Der deutsche Satz {n}."
        return {"choices": [{"text": text}]}


def _patch_llama(monkeypatch_llm):
    """Fake llama_cpp AND torch in sys.modules so translate_segments never
    imports the real torch (slow Windows DLL load / hangs in CI)."""
    import types
    fake = types.ModuleType("llama_cpp")
    fake.Llama = monkeypatch_llm
    fake.llama_supports_gpu_offload = lambda: False
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    return fake, fake_torch


def test_batch_translation_happy_path(tmp_path, monkeypatch):
    import types, sys as _sys
    fake, fake_torch = _patch_llama(MockLlama)
    _sys.modules["llama_cpp"] = fake
    _sys.modules["torch"] = fake_torch
    monkeypatch.setattr(ud, "log", lambda *a, **k: None)  # silence

    # model file must exist for the code path
    model_file = tmp_path / "fake.gguf"
    model_file.write_bytes(b"x")

    segs = _make_segments(6)  # 2 batches of 3
    out = ud.translate_segments(segs, str(model_file), "en", "de", batch_size=3)
    assert all(s.translated_text.startswith("Der deutsche Satz") for s in out), \
        [s.translated_text for s in out]
    # batch path: mock numbers lines 1..N locally per batch
    assert out[0].translated_text == "Der deutsche Satz Nummer 1."   # batch 1, line 1
    assert out[2].translated_text == "Der deutsche Satz Nummer 3."   # batch 1, line 3
    assert out[3].translated_text == "Der deutsche Satz Nummer 1."   # batch 2, line 1
    assert out[5].translated_text == "Der deutsche Satz Nummer 3."   # batch 2, line 3


def test_batch_fallback_on_bad_output(tmp_path, monkeypatch):
    import types, sys as _sys

    class BrokenBatch(MockLlama):
        def __call__(self, prompt, **kw):
            if "numbered" in prompt:
                return {"choices": [{"text": "garbage without numbers"}]}
            return super().__call__(prompt, **kw)

    _sys.modules["llama_cpp"], _sys.modules["torch"] = _patch_llama(BrokenBatch)
    monkeypatch.setattr(ud, "log", lambda *a, **k: None)

    model_file = tmp_path / "fake.gguf"
    model_file.write_bytes(b"x")

    segs = _make_segments(3)
    out = ud.translate_segments(segs, str(model_file), "en", "de", batch_size=3)
    # fallback produced per-segment translations, all valid
    assert all(s.translated_text.startswith("Der deutsche Satz") for s in out)


def test_batch_size_one(tmp_path, monkeypatch):
    import types, sys as _sys
    _sys.modules["llama_cpp"], _sys.modules["torch"] = _patch_llama(MockLlama)
    monkeypatch.setattr(ud, "log", lambda *a, **k: None)

    model_file = tmp_path / "fake.gguf"
    model_file.write_bytes(b"x")

    segs = _make_segments(4)
    out = ud.translate_segments(segs, str(model_file), "en", "de", batch_size=1)
    assert all(s.translated_text.startswith("Der deutsche Satz") for s in out)
