# Censor Module MVP

First implementation of a multi-layer image guardrail for `text2image`, `img2img stylization`, and `img2img editing`.

## What is implemented

- unified moderation API for input and output checks
- text guard stub that always marks prompt text as safe
- OCR adapter for text found inside images
- visual multi-label image classifier adapter
- explicit-content detector adapter
- policy-judge layer with a heuristic fallback and an optional ShieldGemma integration point
- rule-based decision engine with `allow / review / block`
- CLI for local moderation runs

## Architecture

```text
request
  -> text guard stub
  -> image fast analyzer
       -> OCR
       -> visual classifier
       -> explicit content detector
  -> policy judge
  -> decision engine
  -> verdict + category + rationale
```

## Quick start

1. Create an environment with Python 3.11+.
2. Install the base dependencies:

```bash
pip install -r requirements.txt
```

3. If you want the visual ML adapters too, install the extra ML dependencies:

```bash
pip install -r requirements-ml.txt
```

4. Install the Tesseract binary for your OS:

- Windows: install Tesseract and make sure `tesseract.exe` is available, for example under `C:\Program Files\Tesseract-OCR\tesseract.exe`
- macOS: `brew install tesseract`
- Linux: install the system package, for example `sudo apt install tesseract-ocr`

The repository already contains `tools/tessdata/eng.traineddata` and `tools/tessdata/rus.traineddata`, so Russian OCR does not depend on system language packs.

5. Run the API:

```bash
uvicorn censor_guard.app:app --reload
```

6. Open the docs:

`http://127.0.0.1:8000/docs`

## CLI example

```bash
python -m censor_guard.cli --scenario output --image-path path/to/image.png
```

## Environment variables

- `CENSOR_ENABLE_OCR=true|false`
- `CENSOR_ENABLE_VISUAL_CLASSIFIER=true|false`
- `CENSOR_ENABLE_EXPLICIT_DETECTOR=true|false`
- `CENSOR_ENABLE_POLICY_JUDGE=true|false`
- `CENSOR_VISUAL_MODEL_ID=openai/clip-vit-base-patch32`
- `CENSOR_EXPLICIT_MODEL_ID=Falconsai/nsfw_image_detection`
- `CENSOR_POLICY_JUDGE_MODEL_ID=google/shieldgemma-2-4b-it`
- `CENSOR_BLOCK_THRESHOLD=0.85`
- `CENSOR_REVIEW_THRESHOLD=0.55`
- `CENSOR_HF_CACHE_DIR=D:/alpha_siirius/censor_module/.cache/huggingface`
- `CENSOR_TESSERACT_CMD=D:/alpha_siirius/censor_module/tools/Tesseract-OCR/tesseract.exe`

OCR lookup order:
- project-local `tools/tessdata`, if present
- `CENSOR_TESSERACT_CMD`, if set
- `tesseract` from `PATH`
- Windows defaults under `C:\Program Files\Tesseract-OCR`
- macOS defaults such as `/opt/homebrew/bin/tesseract` and `/usr/local/bin/tesseract`
- Linux default `/usr/bin/tesseract`

OCR tries `rus`, `eng`, and `rus+eng`. If `tools/tessdata` exists in the project, it is used as the preferred source of language data.

To make Russian OCR work on another machine, commit both:
- `tools/tessdata/eng.traineddata`
- `tools/tessdata/rus.traineddata`

## Repository structure

```
img-censorship-module/
│
├── censor_guard/               # Весь основной код модуля
│   ├── app.py                  # HTTP API: принимает запрос, возвращает вердикт
│   ├── cli.py                  # То же самое, но запускается из командной строки
│   ├── pipeline.py             # Запускает все проверки по очереди и собирает результат
│   ├── decision.py             # Смотрит на результаты проверок и выносит итог: allow / review / block
│   ├── schemas.py              # Описание формата запроса и ответа
│   ├── taxonomy.py             # Список категорий нарушений (что блокируем жёстко, что — мягко)
│   ├── config.py               # Настройки: какие проверки включены, пороги, пути к моделям
│   ├── image_utils.py          # Вспомогательная функция загрузки картинки
│   ├── evaluation.py           # Утилита для тестирования качества на датасете
│   └── adapters/               # Отдельные проверки — каждая независима, все можно отключить
│       ├── visual_classifier.py    # Смотрит на картинку и определяет, к какой категории она относится (CLIP)
│       ├── explicit_detector.py    # Специализированная проверка на NSFW-контент
│       ├── ocr.py                  # Вытаскивает текст, который написан прямо на картинке
│       ├── text_stub.py            # Проверка текстового промпта — пока заглушка, всегда пропускает
│       └── policy_judge.py         # Финальный арбитр: объединяет сигналы от всех проверок
│
├── ui/                         # Прототип веб-интерфейса
│   ├── app.py                  # Сервер с тем же API + отдаёт HTML-страницу
│   ├── static/                 # Файлы фронтенда (HTML, JS)
│   └── tests/
│       └── test_app.py
│

├── tools/                      # Вспомогательное, не нужно для запуска сервиса
│   ├── tesseract-installer.exe         # Установщик Tesseract для Windows
│   └── tessdata/                       # Языковые файлы для OCR (рус + англ), чтобы не зависеть от системы
│
├── edge_cases/                 # Примеры и заметки по граничным случаям
│
├── censor_module/              # Старая копия пакета, больше не используется
│
├── ARCHITECTURE.md             # Как устроен модуль
├── PLAN.md                     # Что планируется доделать
└── README.md                   # Этот файл
```

Как обрабатывается один запрос:

```
Входящий запрос (картинка + промпт)
  ├── проверка промпта          — пока заглушка, всегда пропускает
  ├── OCR                       — достаёт текст с картинки, если есть
  │    └── проверка OCR-текста  — тоже заглушка
  ├── визуальный классификатор  — к каким категориям относится картинка
  ├── NSFW-детектор             — специализированная проверка на откровенный контент
  └── арбитр                   — смотрит на все сигналы и выставляет итоговые оценки
       └── движок решений      — сравнивает оценки с порогами → allow / review / block
```

## Notes

- The text filter is a placeholder by design. It always returns `safe`, but the adapter contract is already in place.
- The OCR, visual classifier, explicit detector, and ShieldGemma judge are optional adapters. If a backend is unavailable, the service returns a structured `skipped` status instead of failing the whole request.
- The heuristic policy judge keeps the pipeline operational before a real multimodal judge is attached.
