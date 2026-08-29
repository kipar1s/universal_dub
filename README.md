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
:: английский → немецкий (Qwen 0.6B — быстро)
.venv\Scripts\python.exe universal_dub.py video.mp4 --source-lang en --target-lang de

:: английский → немецкий, качество выше (Qwen 1.7B, ~x2 медленнее)
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang de --qwen-size 1.7b

:: VOICE CLONE — клонировать голоса актёров оригинала (лучший вариант!)
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang de --voice-clone --qwen-size 1.7b

:: английский → русский
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang ru

:: украинский (Edge-TTS fallback, Qwen не говорит по-украински)
.venv\Scripts\python.exe universal_dub.py video.mp4 --target-lang uk
```

## Voice Clone (Qwen3-TTS Base)

`--voice-clone` клонирует голоса из оригинала видео:
1. Диаризация разводит реплики по спикерам
2. Для каждого спикера выбирается лучший фрагмент (2–12 с, содержательный текст)
3. Qwen3-TTS-Base клонирует голос по этому фрагменту и озвучивает весь перевод

Результат: дубляж звучит голосами актёров оригинала (на немецком/русском/...).
Нужны: Base-модель (`models/Qwen3-TTS-1.7B-Base`, ~3.5 GB) + минимум ~3 с речи каждого спикера.

## Модели

| Модель | Размер | Назначение | Статус |
|--------|--------|------------|--------|
| Qwen3-TTS-12Hz-0.6B-CustomVoice | ~1.3 GB | быстрое озвучивание (9 спикеров) | ✅ скачана |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | ~3.5 GB | качество выше | ✅ скачана |
| Qwen3-TTS-12Hz-1.7B-Base | ~3.5 GB | voice clone | ✅ скачана |

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

## Быстродействие

### Batch-перевод
По умолчанию 3 сегмента переводятся одним промптом (вместо 3 отдельных LLM-вызовов).
Если LLM нарушил нумерацию/формат — автоматический fallback на поштучный перевод.

```bat
--batch-size 3     :: default
--batch-size 1     :: по одному сегменту (максимальная надёжность)
```

### CUDA-сборка llama-cpp (GPU-перевод, x3-x5 быстрее)
В этом окружении уже собрана с CUDA 13.3. Для повторной сборки:

```bat
G:\universal_dub\build_llama_cuda.bat
```

Требования: MSVC 2022 (VS Community), CUDA Toolkit, cmake 3.31 (4.x ломает
определение CUDA-компилятора на Windows). Проверка:

```bat
.venv\Scripts\python.exe -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
:: True = GPU offload работает
```

## Дорожная карта

- [x] Voice clone (Qwen3-TTS-12Hz-Base) — клонировать голоса актёров оригинала
- [x] 1.7B модель (качество выше) — `--qwen-size 1.7b`
- [x] CUDA-сборка llama-cpp — перевод 8.6x быстрее (0.19 c/сегмент, RTX 3090)
- [x] Batch-перевод — 3 сегмента одним промптом, авто-fallback
- [ ] A/B выбор голоса (auto-select по native language)
- [ ] Web UI (Gradio/Streamlit)
- [ ] Прогресс-бар (rich)
