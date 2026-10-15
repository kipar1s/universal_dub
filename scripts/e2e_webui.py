"""E2E smoke test: submit a real dubbing job through the running Gradio Web UI.

Prereq: `python webui.py` running on http://127.0.0.1:7860 and a real video
at E2E_VIDEO (edit below). Requires gradio_client.
"""
import os
import sys
from gradio_client import Client

E2E_VIDEO = os.environ.get("E2E_VIDEO", r"C:\work\universal_dub\input\e2e_clip.mp4")

c = Client("http://127.0.0.1:7860", verbose=False)
print("connected OK")

# inputs order must match run_btn.click(inputs=[...]) in webui.py:
# file_in, local_path, target, source, engine, qwen_size, voice_clone,
# male_voice, female_voice, batch_size, keep_original, whisper_model, resume_dir
job = c.submit(
    None,                 # file_in (upload) — unused, local path below
    E2E_VIDEO,            # local path
    "de",                 # target lang
    "auto",               # source lang
    "edge",               # engine
    "0.6b",               # qwen size
    False,                # voice clone
    "",                   # male voice (default)
    "",                   # female voice (default)
    3,                    # translation batch size
    False,                # keep original audio
    "tiny",               # whisper model (light on GPU)
    "",                   # resume dir (fresh run)
)
print("submitted OK")

result = job.result(timeout=1200)
log_html, stages_html, video_out = result[0], result[1], result[2]
print("\n===== RESULT =====")
print(video_out)
sys.exit(0 if video_out else 1)
