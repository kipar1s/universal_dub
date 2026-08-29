# universal_dub — Универсальный локальный дубляж видео

Любой из **100 языков Whisper** (исходник) → любой поддерживаемый TTS язык (цель).

```
Video → Audio (ffmpeg)
  → Whisper (100 языков, CUDA)
    → Диаризация (F0 + GMM: муж/жен голоса)
      → Hy-MT 1.5 (LLM-перевод, 28 языков, any→any)
        → TTS: Qwen3-TTS (локально, GPU, 10 языков)
               или Edge-TTS (cloud, 75 языков)
          → Микс (atempo 0.9–1.2x) + Video → Final dubbed video
```

## Почему Qwen3-TTS

Выбран как основной TTS:
- **10 языков**: zh, en, ja, ko, de, fr, ru, pt, es, it
- **9 спикеров** (муж/жен), любой спикер говорит на всех 10 языках
- Локально, GPU (RTX 3090: ~10–12 сек на сегмент, 12 сегментов = ~2.5 мин)
- Качество ⭐⭐⭐ (лучше Silero v5)
- 0.6B модель (~1.3 GB) — быстрая загрузка, 1.7B — качество выше
- Voice clone / voice design — в планах расширения

Для языков вне списка (uk, pl, tr, ar, hi, vi, nl, cs, …) — авто-fallback на Edge-TTS (75 языков, нужен интернет).

## Установка

```bat
cd G:\universal_dub
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu126
```

Модель Qwen3-TTS (скачается в `models/`, ~1.3 GB):
```bat
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', local_dir='G:/universal_dub/models/Qwen3-TTS-0.6B-CustomVoice')"
```

## Запуск

```bat
:: английский → немецкий (Qwen)
.venv\Scripts\python.exe universal_dub.py video.mp4 --source-lang en --target-lang de

:: английский → русский
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang ru

:: украинский (Edge-TTS fallback, Qwen не говорит по-украински)
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang uk

:: свой голос / отключить диаризацию
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang de --male-voice Dylan --female-voice Serena
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang ru --no-diarization
```

Флаг `--source-lang` опционален (автоопределение Whisper), но рекомендуется для
качества перевода (Hy-MT получает явный язык в промпте).

## Языки

| Целевой | Qwen3-TTS (локальный) | Edge-TTS (cloud) |
|---------|:----:|:----:|
| zh, en, ja, ko | ✅ | ✅ |
| de, fr, ru, pt, es, it | ✅ | ✅ |
| uk, pl, tr, ar, hi, vi, nl, cs, sv, el, fi, da, he, th, id, ms, ro, hu, fa, bg | — | ✅ |

Исходный: любой из 100 языков Whisper (транскрипция), перевод Hy-MT — 28 основных.

## Голоса Qwen3-TTS (0.6B/1.7B-CustomVoice)

| Спикер | Описание | Родной язык |
|--------|----------|-------------|
| Vivian | Ярная молодая женская | zh |
| Serena | Тёплая мягкая женская | zh |
| Ono_Anna | Игривая японская | ja |
| Sohee | Тёплая корейская | ko |
| Dylan | Молодой пекинский мужской | zh |
| Eric | Энергичный чэнду | zh |
| Uncle_Fu | Низкий мужской | zh |
| Ryan | Динамичный мужской | en |
| Aiden | Солнечный американский | en |

По умолчанию: мужской → `Uncle_Fu`, женский → `Vivian` (для всех языков).
Нативные: zh→Serena/Dylan, en→Vivian/Ryan, ja→Ono_Anna/Dylan, ko→Sohee/Dylan.

## Папки

```
universal_dub/
├── universal_dub.py   # пайплайн (единый файл)
├── requirements.txt
├── models/Qwen3-TTS-0.6B-CustomVoice/   # модель (~1.3 GB)
└── output/
    ├── <video>_<lang>_dubbed.mp4
    └── artifacts/
        ├── *_audio.wav, *_transcript.json, *_translated.json
        ├── tts_segments/   (сгенерированные wav)
        └── temp_tts_stretch/
```

## Известные ограничения

- **Темп озвучки**: clamp 0.9x–1.2x от естественной TTS (если перевод короче оригинала на >12% — тишина, не замедляется дальше 0.9x)
- **Qwen3-TTS 0.6B**: скорость ~10–12 с/сегмент; 1.7B — качество выше, но медленнее
- **Hy-MT CPU**: без CUDA-сборки llama-cpp перевод ~5 с/сегмент (12 сегментов = ~1 мин)
- **Диаризация**: F0-порог 160 Гц (мужской < 160); может путать очень низкие женские / высокие мужские голоса
- **Edge-TTS**: иногда отдаёт пустой ответ (2 из 12 в тесте) — сегмент остаётся без озвучки
- **Voice clone**: не подключено (модель Qwen3-TTS-12Hz-Base, ~5 GB) — в планах
- **Перевод uk**: Hy-MT путает ru/uk — для uk-оригинала ставьте `--source-lang uk`

## Дорожная карта

- [ ] Voice clone (Qwen3-TTS-12Hz-Base) — клонировать голоса актёров оригинала
- [ ] 1.7B модель (качество выше)
- [ ] CUDA-сборка llama-cpp (перевод x5 быстрее)
- [ ] Batch-перевод (один промпт на 5–10 сегментов)
- [ ] A/B выбор голоса (auto-select по native language)
- [ ] Web UI (Gradio/Streamlit)
- [ ] Прогресс-бар (rich)
