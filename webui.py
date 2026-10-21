#!/usr/bin/env python3
"""
universal_dub — Web UI (Gradio), русская версия

Запуск:  python webui.py          → http://127.0.0.1:7860
         WEBUI_PORT=9000 python webui.py   → свой порт

Особенности:
  - спокойная тёмная тема, живой трекер этапов (извлечение → … → сборка)
  - загрузка видео файлом ИЛИ локальный путь Windows
  - языки с понятными названиями («Русский — Россия» и т.д.)
  - движок TTS (авто/qwen/chatterbox/edge), размер Qwen
  - клонирование голосов + именованные мужской/женский голоса
  - УСТОЙЧИВОСТЬ: пайплайн работает в собственном потоке и НЕ зависит
    от соединения браузера. Если вкладка отвалилась — процесс продолжается,
    повторное нажатие «Озвучить» подключается к уже идущему заданию.
  - результат всегда сохраняется в папку вывода (можно дождаться/дорезюмить)
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# Windows-консоль (cp866/cp1251) не кодирует часть Unicode — не падать на print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# Слабая сеть: дольше ждать HF, не спамить предупреждениями про симлинки.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

import gradio as gr  # noqa: E402
import universal_dub as ud  # noqa: E402

PORT = int(os.environ.get("WEBUI_PORT", "7860"))
OUT_ROOT = HERE / "webui_output"

# ─── Понятные названия языков для интерфейса (код → «Язык — Страна») ──
LANG_DISPLAY_RU = {
    "af": "Африкаанс — ЮАР",
    "ar": "Арабский — Саудовская Аравия",
    "bg": "Болгарский — Болгария",
    "bn": "Бенгальский — Бангладеш",
    "ca": "Каталанский — Испания",
    "cs": "Чешский — Чехия",
    "da": "Датский — Дания",
    "de": "Немецкий — Германия",
    "el": "Греческий — Греция",
    "en": "Английский — Великобритания / США",
    "es": "Испанский — Испания",
    "fa": "Персидский — Иран",
    "fi": "Финский — Финляндия",
    "fr": "Французский — Франция",
    "he": "Иврит — Израиль",
    "hi": "Хинди — Индия",
    "hu": "Венгерский — Венгрия",
    "id": "Индонезийский — Индонезия",
    "it": "Итальянский — Италия",
    "ja": "Японский — Япония",
    "kk": "Казахский — Казахстан",
    "ko": "Корейский — Корея",
    "mr": "Маратхи — Индия",
    "ms": "Малайский — Малайзия",
    "nl": "Нидерландский — Нидерланды",
    "pl": "Польский — Польша",
    "pt": "Португальский — Бразилия",
    "ro": "Румынский — Румыния",
    "ru": "Русский — Россия",
    "sv": "Шведский — Швеция",
    "ta": "Тамильский — Индия",
    "th": "Тайский — Таиланд",
    "tr": "Турецкий — Турция",
    "uk": "Украинский — Украина",
    "ur": "Урду — Пакистан",
    "vi": "Вьетнамский — Вьетнам",
    "zh": "Китайский — Китай",
}


def _display(code: str) -> str:
    return LANG_DISPLAY_RU.get(code, ud.LANG_NAMES.get(code, code))


TARGET_LANGS = sorted(set(ud.EDGE_VOICES) | set(ud.QWEN_LANGUAGES),
                     key=lambda k: _display(k))
TARGET_CHOICES = [(_display(k), k) for k in TARGET_LANGS]

SOURCE_CODES = ["auto"] + sorted(ud.LANG_NAMES.keys(), key=lambda k: _display(k))
SOURCE_CHOICES = [("Авто — определить автоматически", "auto")] + \
                 [(_display(k), k) for k in SOURCE_CODES[1:]]

# ─── Состояние задания (живёт НЕЗАВИСИМО от соединения браузера) ──────
# Если SSE/вебсокет Gradio отвалится на слабом ПК — поток пайплайна
# продолжит работу и допишет файлы в папку вывода. Повторное нажатие
# кнопки подключается к текущему заданию вместо отказа.
_job_lock = threading.Lock()
_job = None  # dict или None


def _new_job(out_dir):
    return {
        "logs": [],            # [(line, level)]
        "stages": {},          # stage -> (done, total)
        "result": None,
        "status": "running",   # running | done | error
        "error": None,
        "out_dir": str(out_dir),
        "started": time.time(),
    }


def _log_to_job(job, msg, level="INFO"):
    line = f"[{datetime.now():%H:%M:%S}] [{level}] {msg}"
    job["logs"].append((line, level))


def _progress_cb(stage, done, total):
    if _job is not None:
        _job["stages"][stage] = (done, total)


def _pipeline_worker(job, path, source_lang, target_lang, engine, qwen_size,
                     voice_clone, male_voice, female_voice, batch_size,
                     keep_original, whisper_model):
    """Работает в своём потоке; UI только читает job — отключение клиента не мешает."""
    try:
        out = ud.run_pipeline(
            video_path=path,
            source_lang=(source_lang or "auto").lower() if source_lang != "auto" else None,
            target_lang=(target_lang or "").lower() or None,
            output_dir=job["out_dir"],
            tts_engine=engine,
            qwen_size=qwen_size,
            voice_clone=bool(voice_clone),
            male_voice=(male_voice or "").strip() or None,
            female_voice=(female_voice or "").strip() or None,
            batch_size=int(batch_size),
            remove_original=not keep_original,
            whisper_model=whisper_model or "medium",
        )
        job["result"] = out
        job["status"] = "done"
        _log_to_job(job, f"✓ ГОТОВО → {out}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        _log_to_job(job, f"✗ ОШИБКА: {job['error']}", "ERROR")
    finally:
        ud.set_progress_callback(None)
        _job_lock.release()


# ─── Трекер этапов ─────────────────────────────────────────────────────
STAGES = [
    ("extract", "Извлечение"),
    ("transcribe", "Расшифровка"),
    ("diarize", "Дикторы"),
    ("translate", "Перевод"),
    ("tts", "Озвучка"),
    ("merge", "Склейка"),
    ("mux", "Сборка"),
]


def _render_stages(state: dict, error: bool = False) -> str:
    chips = []
    for key, label in STAGES:
        if key in state:
            done, total = state[key]
            if done >= total:
                cls, pct = "st-done", ""
            else:
                cls = "st-error" if error else "st-active"
                pct = f" {int(100 * done / max(total, 1))}%"
        else:
            cls, pct = "st-pending", ""
        chips.append(f'<span class="stage-chip {cls}">{label}{pct}</span>')
    sep = '<span class="stage-sep">›</span>'
    return '<div class="stage-row">' + sep.join(chips) + "</div>"


def _fmt_logs(logs: list) -> str:
    out = []
    for line, level in logs[-300:]:
        cls = {"WARN": "lg-warn", "ERROR": "lg-err"}.get(level, "lg-info")
        out.append(f'<span class="{cls}">{line}</span>')
    return "\n".join(out)


def run_job(video_file, local_path, target_lang, source_lang, engine,
            qwen_size, voice_clone, male_voice, female_voice, batch_size,
            keep_original, whisper_model, resume_dir):
    """Генератор: читает состояние задания. Само задание живёт в своём потоке."""
    global _job

    # Уже есть работающее задание → НЕ отказываем, а подключаемся к нему
    # (защита от «connection lost» на слабом ПК / упавшей вкладке).
    if _job is not None and _job["status"] == "running":
        yield (_fmt_logs(_job["logs"] + [("↻ Подключено к работающему заданию. "
                                          "Папка вывода: " + _job["out_dir"], "INFO")]),
               _render_stages(_job["stages"]), None)
        while _job is not None and _job["status"] == "running":
            time.sleep(1.0)
            yield _fmt_logs(_job["logs"]), _render_stages(_job["stages"]), _job["result"]
        if _job is not None:
            err = _job["status"] == "error"
            yield (_fmt_logs(_job["logs"]),
                   _render_stages(_job["stages"], error=err), _job["result"])
        return

    path = (local_path or "").strip()
    if not path and video_file:
        path = video_file[0] if isinstance(video_file, list) else str(video_file)
    if not path:
        yield "✗ Не указан видеофайл — загрузите файл или введите локальный путь.", _render_stages({}), None
        return
    if not os.path.isfile(path):
        yield f"✗ Файл не найден:\n{path}", _render_stages({}), None
        return

    if not _job_lock.acquire(blocking=False):
        yield "⏳ Задание только что стартовало с другого окна. Подождите пару секунд и нажмите ещё раз — подключусь к нему.", _render_stages({}), None
        return

    rd = (resume_dir or "").strip()
    if rd:
        out_dir = Path(rd).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = OUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(path).stem}"

    job = _new_job(out_dir)
    _job = job
    _log_to_job(job, f"Папка вывода: {out_dir}")
    if rd:
        _log_to_job(job, "РЕЖИМ ПРОДОЛЖЕНИЯ: сохранённые расшифровка/перевод/озвучка будут переиспользованы")
    ud.set_progress_callback(_progress_cb)

    t = threading.Thread(
        target=_pipeline_worker,
        args=(job, path, source_lang, target_lang, engine, qwen_size,
              voice_clone, male_voice, female_voice, batch_size,
              keep_original, whisper_model),
        daemon=True,
    )
    t.start()

    # Опрос состояния раз в секунду (щадящий режим для слабого ПК).
    while job["status"] == "running":
        time.sleep(1.0)
        yield _fmt_logs(job["logs"]), _render_stages(job["stages"]), job["result"]

    err = job["status"] == "error"
    stages = dict(job["stages"])
    if not err:
        for key, _ in STAGES:
            total = stages.get(key, (1, 1))[1]
            stages[key] = (total, total)
    yield _fmt_logs(job["logs"]), _render_stages(stages, error=err), job["result"]


# ─── Интерфейс ─────────────────────────────────────────────────────────

CSS = """
:root, .dark {
  --body-background-fill: #171b20;
  --background-fill-primary: #1d232a;
  --background-fill-secondary: #232a33;
  --block-background-fill: #1f262e;
  --block-border-color: #323b46;
  --block-label-text-color: #93a0ad;
  --body-text-color: #d7dde4;
  --body-text-color-subdued: #93a0ad;
  --input-background-fill: #191f26;
  --input-border-color: #3a4450;
  --input-border-color-focus: #6f8ba8;
  --link-text-color: #8fa9c4;
  --checkbox-background-color: #5f7d99;
  --radio-button-background-fill: #191f26;
  --button-primary-background-fill: #4f6f8c;
  --button-primary-background-fill-hover: #5d7f9e;
  --button-primary-text-color: #f0f4f8;
}
.gradio-container { background: #171b20 !important; }
.app-header h1 { letter-spacing: .3px; font-weight: 600; }
.app-header p { color: #93a0ad; }
.stage-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
             padding: 10px 12px; background: #1d232a; border: 1px solid #323b46;
             border-radius: 10px; }
.stage-sep { color: #46525f; }
.stage-chip { padding: 4px 11px; border-radius: 999px; font-size: 13px;
              font-weight: 500; border: 1px solid transparent; white-space: nowrap; }
.st-pending { color: #6b7684; background: #1b2129; border-color: #2c3540; }
.st-active  { color: #c9b98a; background: #262418; border-color: #4d4630; }
.st-done    { color: #93c2a4; background: #18241d; border-color: #2e4636; }
.st-error   { color: #cf9a9a; background: #261a1a; border-color: #4d3030; }
.logbox textarea, .logbox .output textarea {
  font-family: Consolas, "Cascadia Mono", monospace !important;
  font-size: 12px !important; line-height: 1.45 !important; }
.lg-warn { color: #c9b98a; }
.lg-err  { color: #cf9a9a; }
.lg-info { color: #9fb0c0; }
footer { display: none !important; }
"""

with gr.Blocks(title="universal_dub — озвучка видео") as demo:
    with gr.Column(elem_classes=["app-header"]):
        gr.Markdown(
            "## Озвучка видео · universal_dub\n"
            "Whisper → перевод → Qwen3-TTS / Edge-TTS → сборка. "
            "Процесс не прерывается при потере соединения — повторное "
            "нажатие подключится к текущему заданию."
        )
        stages_html = gr.HTML(_render_stages({}))

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=340):
            with gr.Group():
                file_in = gr.File(label="Файл видео", file_count="single", type="filepath")
                path_in = gr.Textbox(
                    label="или локальный путь к видео (приоритет)",
                    placeholder=r"F:\Downloads\video\clip.mp4",
                )
            with gr.Group():
                tgt = gr.Dropdown(TARGET_CHOICES, value="ru",
                                  label="Целевой язык (на какой озвучиваем)", interactive=True)
                src = gr.Dropdown(SOURCE_CHOICES, value="auto",
                                  label="Исходный язык (на каком говорит видео)")
                eng = gr.Radio(["auto", "qwen", "chatterbox", "edge"], value="auto",
                               label="Движок озвучки (авто = Qwen, если язык поддерживается)")
            with gr.Accordion("Голоса", open=False):
                clone = gr.Checkbox(False, label="Клонировать голоса дикторов оригинала (Qwen / Chatterbox)")
                with gr.Row():
                    male_v = gr.Textbox(
                        label="Мужской голос (имя)", value="",
                        placeholder="Uncle_Fu · ru-RU-DmitryNeural",
                    )
                    female_v = gr.Textbox(
                        label="Женский голос (имя)", value="",
                        placeholder="Vivian · ru-RU-SvetlanaNeural",
                    )
            with gr.Accordion("Дополнительно", open=False):
                qsize = gr.Radio(["0.6b", "1.7b"], value="0.6b", label="Размер модели Qwen3-TTS")
                batch = gr.Slider(1, 8, value=3, step=1,
                                  label="Размер батча перевода (сегментов на запрос)")
                wmodel = gr.Dropdown(["tiny", "base", "small", "medium", "large"],
                                     value="medium", label="Модель Whisper")
                keep_orig = gr.Checkbox(False, label="Оставить оригинальный звук фоном (35%)")
                resume_in = gr.Textbox(
                    label="Продолжить из папки (переиспользовать расшифровку/перевод/озвучку)",
                    placeholder=r"C:\work\universal_dub\webui_output\<папка_прогона>",
                )
            run_btn = gr.Button("Озвучить", variant="primary", size="lg")

        with gr.Column(scale=3):
            log_box = gr.HTML("", elem_classes=["logbox"])
            video_out = gr.Video(label="Результат")

    run_btn.click(
        fn=run_job,
        inputs=[file_in, path_in, tgt, src, eng, qsize, clone, male_v, female_v,
                batch, keep_orig, wmodel, resume_in],
        outputs=[log_box, stages_html, video_out],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=PORT, inbrowser=True,
                css=CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
