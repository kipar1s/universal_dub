# INSTALL_PITFALLS.md — полный гайд по установке universal_dub на Windows

> Этот документ — летопись всех ошибок, которые реально были найдены и починены
> на двух разных компьютерах (слабый dev-ноутбук без NVIDIA и мощная машина).
> Написан для того, чтобы нейросеть/инженер на НОВОМ компьютере прошёл этот путь
> за один заход, а не за неделю мучений.
>
> Репозиторий: https://github.com/kipar1s/universal_dub.git, рабочая ветка **vers-1**.

---

## 0. Быстрый старт (если делать всё правильно с самого начала)

```powershell
# 1. Клон (ветка vers-1, НЕ main!)
git clone https://github.com/kipar1s/universal_dub.git
cd universal_dub
git checkout vers-1

# 2. Виртуальное окружение — ОБЯЗАТЕЛЬНО из venv, а не системный Python
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Сначала torch под ВАШЕ железо (см. раздел 6!), потом остальное
#    NVIDIA (CUDA 12.x):
venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#    или CPU-only:
# venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Зависимости проекта
venv\Scripts\pip install -r requirements.txt

# 5. КРИТИЧНО: проверить, что правильный whisper (см. раздел 1)
venv\Scripts\pip uninstall -y whisper
venv\Scripts\pip install openai-whisper

# 6. Проверка окружения (раздел 9) — до первого запуска!
venv\Scripts\python.exe -c "import whisper; print(whisper.__file__)"
# должно быть: ...\venv\Lib\site-packages\whisper\__init__.py (ПАПКА!), а НЕ whisper.py (файл)

# 7. Путь к модели перевода Hy-MT (раздел 5) — задать ДО запуска
set UD_HY_MT_MODEL=D:\models\HY-MT1.5-1.8B-Q4_K_M.gguf

# 8. Запуск
venv\Scripts\python.exe webui.py
# открыть http://127.0.0.1:7860/
```

---

## 1. ГРАБЛИ №1 (самые злые): `TypeError: argument of type 'NoneType' is not iterable`

**Симптом:** запуск `webui.py`, всё доходит до `Audio extracted: ...wav`, и сразу:
```
✗ ERROR: TypeError: argument of type 'NoneType' is not iterable
```
Никакого упоминания whisper в логе. Ошибка выглядит как «кончилась видеопамять» — **это НЕ она**.

**Истинная причина:** в venv стоит пакет **`whisper` (graphite, версия 1.1.x)** вместо
**`openai-whisper`**. Это два РАЗНЫХ пакета с одним именем импорта. Graphite-версия
крашится уже при `import whisper`:
```
venv\Lib\site-packages\whisper.py, line 69:
    libc = ctypes.CDLL(libc_name)     # libc_name == None
TypeError: argument of type 'NoneType' is not iterable
```
Обратите внимание: `whisper.py` (файл) = graphite (плохо),
`whisper\__init__.py` (папка) = openai-whisper (хорошо).

**Как так вышло:** `pip install whisper` ставит graphite-пакет. openai-whisper
ставится ТОЛЬКО как `pip install openai-whisper`.

**Фикс:**
```powershell
venv\Scripts\pip uninstall -y whisper
venv\Scripts\pip install openai-whisper
```

**Диагностика одной строкой:**
```powershell
venv\Scripts\python.exe -c "import whisper; print(whisper.__file__)"
```

В коде (`universal_dub.py: transcribe_audio`) теперь стоит try/except вокруг
`import whisper` с человекочитаемым сообщением — если увидите его в логе, делайте фикс выше.

**Повторяемость:** эта ошибка вылезла НА ОБОИХ компьютерах, потому что venv
переносился/пересоздавался и кто-то снова ставил `pip install whisper`.
**Правило: никогда не пишите `pip install whisper`. Только `openai-whisper`.**

---

## 2. ГРАБЛИ №2: Edge-TTS отдаёт HTTP 403 (WSServerHandshakeError)

**Симптом:** этап TTS падает на каждом сегменте:
```
aiohttp.client_exceptions.WSServerHandshakeError: 403, message='Invalid response status'
```

**Причина:** Microsoft поменял consumer-эндпоинт Edge-TTS; старые версии
`edge-tts` (7.2.8 и ниже) больше не авторизуются.

**Фикс:**
```powershell
venv\Scripts\pip install --upgrade "edge-tts>=7.3.1"
```
Проверено: 7.3.1 работает. Если снова 403 — обновляйте edge-tts до последней
версии, это чисто версионная поломка.

**Второй нюанс (уже починен в коде):** Edge-TTS пишет **MP3 независимо от
расширения файла**. Если сохранить как `.wav` — получится WAV-файл с mp3-внутри,
и ffmpeg/sox сломаются ниже по пайплайну. В `EdgeTTS.generate_sync()` теперь
стоит конвертация mp3→WAV 24 кГц mono через ffmpeg. Если будете портировать код —
не убирайте этот шаг.

---

## 3. ГРАБЛИ №3: `ModuleNotFoundError: No module named 'qwen_tts'` / `chatterbox`

**Симптом:** выбор движка Qwen или Chatterbox в UI → жёсткий краш.

**Фикс в коде:** `create_tts_engine()` теперь ловит ImportError и делает
WARN + автофолбэк на Edge-TTS.

**Если Qwen нужен по-настоящему:**
```powershell
venv\Scripts\pip install qwen-tts
```
Тянет за собой transformers, accelerate, torchaudio, onnxruntime, gradio —
большая установка, может конфликтовать с уже стоящим gradio (проверьте
`pip show gradio` после).

Модели Qwen3-TTS лежат в `models/` (0.6B-CustomVoice, 1.7B-Base, 1.7B-CustomVoice).
При первом запуске они докачиваются с HuggingFace — см. грабли №7 про таймауты.

**Реальность по железу:**
- Без NVIDIA (Intel UHD / CPU-only torch) Qwen-TTS формально работает, но
  НЕИСПОЛЬЗУЕМО медленно. Для CPU-машины единственный разумный движок — **edge**.
- На мощной машине с NVIDIA убедитесь, что torch именно CUDA-версии (раздел 6).

---

## 4. ГРАБЛИ №4: `llama-cpp-python` не собирается (ошибка компиляции при pip install)

**Симптом:** `pip install llama-cpp-python` падает с CMake/MSVC ошибками —
на Windows без Visual Studio Build Tools сборка из исходников не проходит.

**Фикс (CPU-версия, готовые колёса):**
```powershell
venv\Scripts\pip install llama-cpp-python==0.1.57 `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**Для NVIDIA-машины** — брать CUDA-колёса с того же индекса
(`.../whl/cu124` и т.п., сверяйтесь с документацией abetlen), либо ставить
Build Tools и собирать.

llama-cpp-python нужен для Hy-MT перевода (GGUF). Без него пайплайн работает,
но перевод пропускается.

---

## 5. ГРАБЛИ №5: Hy-MT модель не найдена → озвучка на ЯЗЫКЕ ОРИГИНАЛА

**Симптом (был в живом логе):**
```
[WARN] Hy-MT model not found at 'G:\work\hermes\vvp-translate\.models\HY-MT1.5-1.8B-Q4_K_M.gguf' — using source text as-is
```
Пайплайн «успешно» завершается, но видео озвучено **по-английски** — перевод
просто пропущен. Это НЕ краш, поэтому легко проглядеть.

**Причина:** путь к модели по умолчанию захардкожен под ДРУГОЙ компьютер
(диск `G:`, которого на новом машине нет).

**Фикс:** скачать `HY-MT1.5.5-1.8B-Q4_K_M.gguf` (HuggingFace: Tencent/Hy-MT2)
и задать путь ЛЮБЫМ из:
1. ENV: `set UD_HY_MT_MODEL=D:\models\HY-MT1.5-1.8B-Q4_K_M.gguf`
2. CLI: `--hy-mt-model D:\models\...gguf`
3. Поле в UI (Дополнительно)

**Обязательная проверка после настройки:** в логе прогона НЕ должно быть
строки `using source text as-is`. Если есть — перевода нет, результат не смотреть.

---

## 6. ГРАБЛИ №6: CUDA / «не хватает видеопамяти» / torch CPU-only

**Симптом:** `Using device: cpu` в логе, хотя стоит RTX.

**Причина:** torch установлен по умолчанию с PyPI — это CPU-сборка
(`2.14.0+cpu`). CUDA-версия ставится ТОЛЬКО с другим index-url (см. раздел 0, шаг 3).

**Проверка:**
```powershell
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```
`False` при наличии RTX → переустановить torch с cu121/cu124.

**Реальная нехватка VRAM** выглядит иначе: `CUDA out of memory` в логе.
Тогда: Whisper `small` вместо `medium`/`large`, Qwen 0.6B вместо 1.7B,
закрыть браузер/игры, `--whisper-model tiny` для быстрой проверки.

**Ориентир по времени (замерено на CPU, i5-8гопоколения, 2-мин видео):**
| Этап | tiny | medium |
|---|---|---|
| Whisper | ~10 с | ~100 с |
| Diarization | ~40 с | ~40 с |
| TTS edge (10 сегм.) | ~60 с | — |
| **Итого e2e** | **111 с** | ~5 мин |

Qwen-TTS на CPU — на порядок медленнее. На RTX всё это десятки секунд.

---

## 7. ГРАБЛИ №7: HuggingFace ReadTimeout при скачивании моделей

**Симптом:** прогон висит/падает на загрузке модели, в логе:
```
httpx.ReadTimeout ... huggingface.co
```

**Фиксы (уже вшиты в webui.py):** `HF_HUB_ETAG_TIMEOUT=30`.
Дополнительно на медленном/цензурируемом нете:
```powershell
set HF_ENDPOINT=https://hf-mirror.com
```
Модели Qwen3-TTS лучше скачать ОДИН РАЗ вручную и положить в `models/` —
при повторных запусках сеть не нужна.

---

## 8. ГРАБЛИ №8: обрыв связи браузера убивал задание («потерянная попытка реконнекта»)

**Симптом (был в живом логе):** лог обрывается на `Generating TTS for 3 segments (qwen)…`,
браузер пишет «Connection lost, reconnect failed», всё «умирает».

**Причина (старая версия):** пайплайн жил внутри SSE-потока Gradio —
нет соединения с вкладкой, нет и задания.

**Фикс (в vers-1, коммит ee6c49e):** задание крутится в отдельном
`threading.Thread(daemon=True)` на сервере. Вкладку можно закрыть — прогон
допишет результат в `webui_output\`. Повторное нажатие «Озвучить» подключается
к идущему заданию, а не запускает второе.

**Если после обрыва сомневаетесь, упало ли задание:** смотрите папку
`webui_output\<timestamp>_*` и лог `webui_server.log` — скорее всего всё доделалось.

---

## 9. ГРАБЛИ №9: порт 7860 занят «призраком» старого сервера

**Симптом:**
```
OSError: Cannot find empty port in range: 7860-7860
```
при этом «сервер же не запущен». Или: открывается СТАРЫЙ интерфейс, хотя код
уже новый — потому что на порту висит процесс из прошлой сессии (возможно,
на системном Python, а не venv!).

**Фикс:**
```powershell
netstat -ano | findstr :7860
taskkill /PID <номер> /F
```
Правило: **всегда запускайте через `venv\Scripts\python.exe`**, а не
`python webui.py` — иначе запустится на системном Python без зависимостей.

---

## 10. ГРАБЛИ №10: pytest собирает мусор и падает на коллекте

**Симптом:** `pytest` падает ещё до тестов — root-скрипты `test_*.py`
тянут GPU/сеть при коллекте.

**Фикс (уже в репо):** smoke-скрипты переехали в `scripts/`
(`smoke_transcribe.py`, `smoke_chatterbox.py`, `e2e_webui.py`).
Не возвращайте файлы вида `test_*.py` в корень — только в `scripts/` или `tests/`.
Нормальный прогон: `venv\Scripts\python.exe -m pytest tests/ -q` → **22 passed**.

---

## 11. ГРАБЛИ №11: git push зависает на Windows (credential manager)

**Симптом:** `git push` висит вечно; в GIT_TRACE:
```
error: cannot spawn /mingw64/bin/git-askpass.exe: No such file or directory
bash: line 1: /dev/tty: No such device or address
```
Из неинтерактивной оболочки (агент, cron, IDE-терминал без TTY) push может
недопустить окно логина — и висит.

**Фиксы (по порядку предпочтения):**
1. Выполнить push из **обычного** PowerShell/cmd — появится окно GitHub-логина.
2. Разово: `git push https://<TOKEN>@github.com/kipar1s/universal_dub.git vers-1`
   (PAT с правом repo; не коммитить remote URL с токеном!).
3. `cmdkey /list` — проверить/перезаписать запись git:https://github.com.

**Полезно:** `git ls-remote origin` работает без окна (кэш учётки) — им удобно
ПОДТВЕРЖДАТЬ, что push реально дошёл: хеш в `refs/heads/vers-1` должен совпасть
с локальным `git rev-parse vers-1`.

---

## 12. ГРАБЛИ №12: SoX не установлен

**Симптом:** WARN при клонировании голоса (Qwen voice-clone использует sox
для ресемплинга референсных клипов).

**Фикс:** `winget install --id Rebsox.org_Sox` и/или задать `UD_SOX_DIR`
(ENV) на папку с sox.exe. Без SoX edge-движок работает нормально — нужен он
только для Qwen-клона.

---

## 13. Мелочи, съевшие время

- **ffmpeg**: путь ищется авто-детектом (winget-сборка Gyan.FFmpeg), но если
  ffmpeg в нестандартном месте — `set UD_FFMPEG=C:\path\ffmpeg.exe`. ffprobe
  должен лежать рядом.
- **requirements.txt содержит `whisper`** — если pip проглотит и поставит
  graphite, см. грабли №1. Всегда сверяйте `whisper.__file__` после установки.
- **gradio**: qwen-tts может принести свою версию gradio. UI проверен на
  gradio 6.x. После `pip install qwen-tts` перепроверьте `webui.py` запуском.
- **Edge-голоса**: список `EDGE_VOICES` в `universal_dub.py` сверен живым
  `edge_tts.list_voices()` — не «чините» имена голосов наугад, половина
  «неправильных» на самом деле единственные существующие (например,
  `uk-UA-OstapNeural` вместо выдуманного Pavlo).
- **`.git` раздувался до ~1 ГБ** из-за висячих объектов — лечится
  `git gc --prune=now` (сжал до 7 МБ).

---

## 14. Чек-лист «машина готова к работе» (запустить ВСЁ до первого видео)

```powershell
venv\Scripts\python.exe -c "import whisper; assert 'whisper' in whisper.__file__ and whisper.__file__.endswith('__init__.py'), 'graphite whisper!'"; echo WHISPER_OK
venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
venv\Scripts\python.exe -c "import edge_tts; print('edge-tts', edge_tts.__version__ if hasattr(edge_tts,'__version__') else 'ok')"
venv\Scripts\python.exe -c "from llama_cpp import Llama; print('llama OK')"
ffmpeg -version | findstr version
echo %UD_HY_MT_MODEL%   # должен указывать на существующий .gguf
venv\Scripts\python.exe -m pytest tests/ -q   # 22 passed
```

Затем короткий тестовый прогон (1–2 мин видео, `tiny` + `edge`) и **обязательная
проверка лога**: нет `using source text as-is`, нет `403`, финальная строка
`✓ COMPLETE`. Готово.

---

## 15. Что уже починено в коде (не чините дважды)

| Фикс | Где |
|---|---|
| Понятная ошибка при graphite-whisper | `transcribe_audio()` |
| Fallback Qwen/Chatterbox → Edge при ImportError | `create_tts_engine()` |
| Edge mp3→WAV 24кГц | `EdgeTTS.generate_sync()` |
| «Сохранить оригинал» реально слышен (35%) | `merge_video_audio()` |
| Guard на убегающую длительность TTS | `_tts_duration_bad()` |
| Проверка return-code ffmpeg при растяжении | `merge_tts_segments()` |
| ENV-переопределения путей: `UD_FFMPEG`, `UD_SOX_DIR`, `UD_HY_MT_MODEL` | верх `universal_dub.py` |
| Задание живёт вне SSE-соединения браузера | `webui.py` job runner |
| `HF_HUB_ETAG_TIMEOUT=30` | `webui.py` |
| Валидные Edge-голоса для 30 языков | `EDGE_VOICES` |
| Русский UI + названия «язык — страна» | `webui.py` |

История коммитов: `git log --oneline` (b7f04f8 → f93a951 → e19bc47 → 57933b2 → ee6c49e).
Полный код-ревью: `CODE_REVIEW.md`.
