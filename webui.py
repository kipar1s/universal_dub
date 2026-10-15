#!/usr/bin/env python3
"""
universal_dub — Web UI (Gradio)

Run:  python webui.py          → http://127.0.0.1:7860
      WEBUI_PORT=9000 python webui.py   → custom port

Features:
  - video upload OR local Windows path
  - target/source language, TTS engine (auto/qwen/chatterbox/edge), Qwen size
  - voice-clone toggle + named male/female voices (per-gender cloning on Qwen/Chatterbox)
  - translation batch size, keep-original-audio
  - live log + per-stage progress bar + result player
"""

import os
import sys
import queue
import threading
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# Windows-консоль (cp866/cp1251) не кодирует часть Unicode (→ ✓ ⏳ ✗) —
# не падать на print в любой кодировке консоли.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

import gradio as gr  # noqa: E402
import universal_dub as ud  # noqa: E402

PORT = int(os.environ.get("WEBUI_PORT", "7860"))
OUT_ROOT = HERE / "webui_output"

# ─── Log tee: capture pipeline logs into a queue for the UI ──────
_q: "queue.Queue[tuple]" = queue.Queue()
_orig_log = ud.log


def _tee_log(msg, level="INFO"):
    line = f"[{datetime.now():%H:%M:%S}] [{level}] {msg}"
    _orig_log(msg, level)  # keep console output too
    try:
        _q.put(("log", line))
    except Exception:
        pass


ud.log = _tee_log  # module-level name lookup → all internal calls go through the tee


def _progress_cb(stage, done, total):
    _q.put(("progress", stage, done, total))


# ─── Job runner (one at a time) ──────────────────────────────────
_job_lock = threading.Lock()

TARGET_LANGS = sorted(set(ud.EDGE_VOICES) | set(ud.QWEN_LANGUAGES),
                      key=lambda k: ud.LANG_NAMES.get(k, k))
SOURCE_LANGS = ["auto"] + sorted(ud.LANG_NAMES.keys(),
                                 key=lambda k: ud.LANG_NAMES[k])


def run_job(video_file, local_path, target_lang, source_lang, engine,
            qwen_size, voice_clone, male_voice, female_voice, batch_size,
            keep_original, whisper_model, resume_dir, progress=gr.Progress()):
    """Generator: yields (log_text, video_path) updates until the job finishes."""

    def yield_state(logs, video):
        return "\n".join(logs[-400:]), video

    path = (local_path or "").strip()
    if not path and video_file:
        path = video_file[0] if isinstance(video_file, list) else str(video_file)
    if not path:
        yield "✗ No video specified — upload a file or enter a local path.", None
        return
    if not os.path.isfile(path):
        yield f"✗ File not found:\n{path}", None
        return

    if not _job_lock.acquire(blocking=False):
        yield "⏳ Another job is already running. Wait for it to finish (one GPU, one job).", None
        return

    logs: list[str] = []
    result_video = None
    # drain stale items left by a previous job (client disconnect mid-run)
    while not _q.empty():
        try:
            _q.get_nowait()
        except queue.Empty:
            break
    rd = (resume_dir or "").strip()
    if rd:
        out_dir = Path(rd).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        logs.append(f"RESUME MODE: output dir = {out_dir} "
                    f"(saved transcript/translation/TTS will be reused)")
    else:
        out_dir = OUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(path).stem}"
    try:
        ud.set_progress_callback(_progress_cb)

        def worker():
            try:
                out = ud.run_pipeline(
                    video_path=path,
                    source_lang=(source_lang or "auto").lower() if source_lang != "auto" else None,
                    target_lang=(target_lang or "").lower() or None,
                    output_dir=str(out_dir),
                    tts_engine=engine,
                    qwen_size=qwen_size,
                    voice_clone=bool(voice_clone),
                    male_voice=(male_voice or "").strip() or None,
                    female_voice=(female_voice or "").strip() or None,
                    batch_size=int(batch_size),
                    remove_original=not keep_original,
                    whisper_model=whisper_model or "medium",
                )
                _q.put(("done", out))
            except Exception as e:
                import traceback
                traceback.print_exc()
                _q.put(("error", f"{type(e).__name__}: {e}"))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            try:
                item = _q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = item[0]
            if kind == "log":
                logs.append(item[1])
                yield yield_state(logs, result_video)
            elif kind == "progress":
                _, stage, done, total = item
                try:
                    progress(min(done / max(total, 1), 1.0), desc=f"{stage}: {done}/{total}")
                except Exception:
                    pass
                yield yield_state(logs, result_video)
            elif kind == "done":
                result_video = item[1]
                logs.append(f"✓ DONE → {result_video}")
                break
            elif kind == "error":
                logs.append(f"✗ ERROR: {item[1]}")
                break
    finally:
        _job_lock.release()
        ud.set_progress_callback(None)

    yield yield_state(logs, result_video)


# ─── UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="universal_dub") as demo:
    gr.Markdown(
        "# 🎬 universal_dub\n"
        "**Any language → any language** video dubbing on your GPU.\n"
        "Whisper (transcribe) → Hy-MT 1.8B (translate, CUDA) → Qwen3-TTS / Edge-TTS (dub) → merge."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Video file", file_count="single", type="filepath")
            path_in = gr.Textbox(
                label="…or local video path (takes priority)",
                placeholder=r"F:\Downloads\video\clip.mp4",
            )
            resume_in = gr.Textbox(
                label="Resume from folder (optional — reuse saved transcript/translation/TTS)",
                placeholder=r"C:\work\universal_dub\webui_output\<run_folder>",
            )
            tgt = gr.Dropdown(TARGET_LANGS, value="ru", label="Target language", interactive=True)
            src = gr.Dropdown(SOURCE_LANGS, value="auto", label="Source language")
            eng = gr.Radio(["auto", "qwen", "chatterbox", "edge"], value="auto",
                           label="TTS engine (auto = Qwen if target supported; Chatterbox V3 = 23 langs + cloning)")
            qsize = gr.Radio(["0.6b", "1.7b"], value="0.6b", label="Qwen3-TTS size")
            clone = gr.Checkbox(False, label="Voice-clone original speakers (needs Qwen or Chatterbox engine)")
            with gr.Row():
                male_v = gr.Textbox(
                    label="Male voice name", value="",
                    placeholder="e.g. Uncle_Fu (Qwen/Chatterbox) · en-US-GuyNeural (Edge)",
                )
                female_v = gr.Textbox(
                    label="Female voice name", value="",
                    placeholder="e.g. Vivian (Qwen/Chatterbox) · en-US-AriaNeural (Edge)",
                )
            batch = gr.Slider(1, 8, value=3, step=1,
                              label="Translation batch size (segments per prompt)")
            wmodel = gr.Dropdown(["tiny", "base", "small", "medium", "large"],
                                 value="medium", label="Whisper model")
            keep_orig = gr.Checkbox(False, label="Keep original audio under the dub")
            run_btn = gr.Button("🚀 Dub it", variant="primary")
        with gr.Column(scale=2):
            log_box = gr.Textbox(value="", lines=16, max_lines=16, label="Log (live)")
            video_out = gr.Video(label="Result")

    run_btn.click(
        fn=run_job,
        inputs=[file_in, path_in, tgt, src, eng, qsize, clone, male_v, female_v, batch, keep_orig, wmodel, resume_in],
        outputs=[log_box, video_out],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=PORT, inbrowser=True,
                theme=gr.themes.Soft())
