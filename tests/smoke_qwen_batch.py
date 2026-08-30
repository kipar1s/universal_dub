"""Smoke test: new QwenTTS path (precomputed voice-clone prompt + batched calls)."""
import os, sys, time
sys.path.insert(0, r"G:\universal_dub")
import universal_dub as ud

REFS = {
    "SPEAKER_01": (r"G:\universal_dub\output\artifacts\voice_refs\SPEAKER_01_ref.wav",
                   "You are not going to get any help if you keep saying things like that."),
}

eng = ud.QwenTTS(size="0.6b", kind="base")
t_prep0 = time.time()
eng.set_references(REFS)
t_prep1 = time.time()
print(f"PROMPT_PREP: {t_prep1 - t_prep0:.1f}s, prompts={list(eng.prompts)}")

# 1) single via precomputed prompt
t0 = time.time()
eng.generate_sync("Это проверка нового пути с готовым промптом. Один раз.",
                  r"G:\universal_dub\output\smoke_single.wav",
                  language="Russian", voice="SPEAKER_01")
t1 = time.time()
print(f"SINGLE: {t1 - t0:.1f}s")

# 2) batch of 2 via precomputed prompts
t0 = time.time()
ok, fail = eng.generate_batch_sync(
    [("Раз, два, три — проверка батчинга.", "SPEAKER_01"),
     ("Ещё одна фраза для проверки пакета.", "SPEAKER_01")],
    [r"G:\universal_dub\output\smoke_batch_a.wav",
     r"G:\universal_dub\output\smoke_batch_b.wav"],
    language="Russian", batch_size=8)
t1 = time.time()
print(f"BATCH2: {t1 - t0:.1f}s ok={ok} fail={fail}")

for f in ["smoke_single.wav", "smoke_batch_a.wav", "smoke_batch_b.wav"]:
    p = os.path.join(r"G:\universal_dub\output", f)
    if os.path.exists(p):
        import wave
        with wave.open(p, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        print(f"{f}: {os.path.getsize(p)} bytes, {dur:.2f}s")
    else:
        print(f"{f}: MISSING")
print("SMOKE_OK")
