"""E2E smoke test: submit a real dubbing job through the Gradio Web UI."""
import sys
from gradio_client import Client

c = Client("http://127.0.0.1:7860", verbose=False)
print("connected OK")

job = c.submit(
    None,                                # file_in (upload) — unused
    r"G:\universal_dub\e2e_clip.mp4",    # local path
    "de",                                # target lang
    "auto",                              # source lang
    "edge",                              # engine
    "0.6b",                              # qwen size
    False,                               # voice clone
    3,                                   # batch size
    False,                               # keep original audio
    "tiny",                              # whisper model (light on GPU)
)
print("submitted OK")

result = job.result(timeout=1200)
log_text, video_out = result[0], result[1]
print("\n===== LAST 30 LOG LINES =====")
print("\n".join(log_text.splitlines()[-30:]))
print("\n===== VIDEO OUTPUT =====")
print(video_out)
