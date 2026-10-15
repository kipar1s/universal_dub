#!/usr/bin/env python3
"""
universal_dub — Web UI (Gradio)

Run:  python webui.py          → http://127.0.0.1:7860
      WEBUI_PORT=9000 python webui.py   → custom port

Features:
  - dark studio theme, live pipeline stage tracker (extract → … → mux)
  - video upload OR local Windows path
  - target/source language, TTS engine (auto/qwen/chatterbox/edge), Qwen size
  - voice-clone toggle + named male/female voices (per-gender cloning on Qwen/Chatterbox)
  - translation batch size, keep-original-audio
  - live log + per-stage progress + result player
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
        _q.put(("log", line, level))
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

# ─── Pipeline stage tracker ──────────────────────────────────────
STAGES = [
    ("extract", "🎞", "Extract"),
    ("transcribe", "🎙", "Whisper"),
    ("diarize", "👥", "Speakers"),
    ("translate", "🌐", "Translate"),
    ("tts", "🗣", "TTS"),
    ("merge", "🧩", "Merge"),
    ("mux", "🎬", "Mux"),
]


def _render_stages(state: dict, error: bool = False) -> str:
    """HTML chip-row: pending / active / done per pipeline stage."""
    chips = []
    for key, icon, label in STAGES:
        if key in state:
            done, total = state[key]
            if done >= total:
                cls, pct = "st-done", ""
            else:
                cls = "st-error" if error else "st-active"
                pct = f" {int(100 * done / max(total, 1))}%"
        else:
            cls, pct = "st-pending", ""
        chips.append(
            f'<span class="stage-chip {cls}"><b>{icon}</b> {label}{pct}</span>')
    sep = '<span class="stage-sep">›</span>'
    return '<div class="stage-row">' + sep.join(chips) + "</div>"


def _fmt_logs(logs: list) -> str:
    """Monospace log with level coloring (last 400 lines)."""
    out = []
    for line, level in logs[-400:]:
        cls = {"WARN": "lg-warn", "ERROR": "lg-err"}.get(level, "lg-info")
        out.append(f'<span class="{cls}">{line}</span>')
    return "\n".join(out)


def run_job(video_file, local_path, target_lang, source_lang, engine,
            qwen_size, voice_clone, male_voice, female_voice, batch_size,
            keep_original, whisper_model, resume_dir, progress=gr.Progress()):
    """Generator: yields (log_html, stages_html, video_path) until the job finishes."""

    def yield_state(logs, stages, video, error=False):
        return _fmt_logs(logs), _render_stages(stages, error), video

    path = (local_path or "").strip()
    if not path and video_file:
        path = video_file[0] if isinstance(video_file, list) else str(video_file)
    if not path:
        yield "✗ No video specified — upload a file or enter a local path.", _render_stages({}), None
        return
    if not os.path.isfile(path):
        yield f"✗ File not found:\n{path}", _render_stages({}), None
        return

    if not _job_lock.acquire(blocking=False):
        yield "⏳ Another job is already running. Wait for it to finish (one GPU, one job).", _render_stages({}), None
        return

    logs: list[tuple] = []
    stages: dict = {}
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
        logs.append((f"RESUME MODE: output dir = {out_dir} "
                     f"(saved transcript/translation/TTS will be reused)", "INFO"))
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
                logs.append((item[1], item[2] if len(item) > 2 else "INFO"))
                yield yield_state(logs, stages, result_video)
            elif kind == "progress":
                _, stage, done, total = item
                stages[stage] = (done, total)
                try:
                    progress(min(done / max(total, 1), 1.0), desc=f"{stage}: {done}/{total}")
                except Exception:
                    pass
                yield yield_state(logs, stages, result_video)
            elif kind == "done":
                result_video = item[1]
                for key, _, _ in STAGES:
                    stages.setdefault(key, (1, 1))
                    stages[key] = (stages[key][1], stages[key][1])
                logs.append((f"✓ DONE → {result_video}", "INFO"))
                break
            elif kind == "error":
                logs.append((f"✗ ERROR: {item[1]}", "ERROR"))
                yield yield_state(logs, stages, result_video, error=True)
                break
    finally:
        _job_lock.release()
        ud.set_progress_callback(None)

    yield yield_state(logs, stages, result_video,
                      error=result_video is None)


# ─── UI ──────────────────────────────────────────────────────────

CSS = """
:root, .dark {
  --body-background-fill: #0b0e14;
  --background-fill-primary: #10151f;
  --background-fill-secondary: #161d2a;
  --block-background-fill: #131a26;
  --block-border-color: #232d3f;
  --block-label-text-color: #8b98ad;
  --body-text-color: #dbe4f0;
  --body-text-color-subdued: #8b98ad;
  --input-background-fill: #0d1320;
  --input-border-color: #2a3548;
  --input-border-color-focus: #7c5cff;
  --link-text-color: #9d86ff;
  --checkbox-background-color: #7c5cff;
  --radio-button-background-fill: #0d1320;
}
.gradio-container { background: #0b0e14 !important; }
.app-header h1 { letter-spacing: .5px; }
.app-header p { color: #8b98ad; }
.stage-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
             padding: 10px 12px; background: #10151f; border: 1px solid #232d3f;
             border-radius: 12px; }
.stage-sep { color: #3a465c; }
.stage-chip { padding: 5px 12px; border-radius: 999px; font-size: 13px;
              font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
.st-pending { color: #5b6778; background: #141a26; border-color: #222c3e; }
.st-active  { color: #ffd166; background: #2a230f; border-color: #6b5310;
              animation: pulse 1.2s ease-in-out infinite; }
.st-done    { color: #6ee7a0; background: #0f2318; border-color: #1d4430; }
.st-error   { color: #ff7b7b; background: #2a1010; border-color: #5c1f1f; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .55; } }
.logbox textarea, .logbox .output textarea {
  font-family: Consolas, "Cascadia Mono", monospace !important;
  font-size: 12px !important; line-height: 1.45 !important; }
.lg-warn { color: #ffd166; }
.lg-err  { color: #ff7b7b; }
.lg-info { color: #9fb0c8; }
footer { display: none !important; }
"""

with gr.Blocks(title="universal_dub", css=CSS) as demo:
    with gr.Column(elem_classes=["app-header"]):
        gr.Markdown(
            "# 🎬 universal_dub\n"
            "**Any language → any language** video dubbing on your GPU — "
            "Whisper → Hy-MT 1.8B → Qwen3-TTS / Edge-TTS → merge."
        )
        stages_html = gr.HTML(_render_stages({}))

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=340):
            with gr.Group():
                file_in = gr.File(label="📂 Video file", file_count="single", type="filepath")
                path_in = gr.Textbox(
                    label="…or local video path (takes priority)",
                    placeholder=r"F:\Downloads\video\clip.mp4",
                )
            with gr.Group():
                tgt = gr.Dropdown(TARGET_LANGS, value="ru", label="🎯 Target language", interactive=True)
                src = gr.Dropdown(SOURCE_LANGS, value="auto", label="🗨 Source language")
                eng = gr.Radio(["auto", "qwen", "chatterbox", "edge"], value="auto",
                               label="🔊 TTS engine (auto = Qwen if supported)")
                qsize = gr.Radio(["0.6b", "1.7b"], value="0.6b", label="Qwen3-TTS size")
            with gr.Accordion("🧬 Voice & voices", open=False):
                clone = gr.Checkbox(False, label="Voice-clone original speakers (Qwen / Chatterbox)")
                with gr.Row():
                    male_v = gr.Textbox(
                        label="Male voice name", value="",
                        placeholder="Uncle_Fu · en-US-GuyNeural",
                    )
                    female_v = gr.Textbox(
                        label="Female voice name", value="",
                        placeholder="Vivian · en-US-AriaNeural",
                    )
            with gr.Accordion("⚙️ Advanced", open=False):
                batch = gr.Slider(1, 8, value=3, step=1,
                                  label="Translation batch size (segments per prompt)")
                wmodel = gr.Dropdown(["tiny", "base", "small", "medium", "large"],
                                     value="medium", label="Whisper model")
                keep_orig = gr.Checkbox(False, label="Keep original audio under the dub (35%)")
                resume_in = gr.Textbox(
                    label="Resume from folder (reuse saved transcript/translation/TTS)",
                    placeholder=r"C:\work\universal_dub\webui_output\<run_folder>",
                )
            run_btn = gr.Button("🚀 Dub it", variant="primary", size="lg")

        with gr.Column(scale=3):
            log_box = gr.HTML("", elem_classes=["logbox"])
            video_out = gr.Video(label="Result")

    run_btn.click(
        fn=run_job,
        inputs=[file_in, path_in, tgt, src, eng, qsize, clone, male_v, female_v, batch, keep_orig, wmodel, resume_in],
        outputs=[log_box, stages_html, video_out],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=PORT, inbrowser=True,
                theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"))
