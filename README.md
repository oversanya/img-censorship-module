# Censor Module MVP

Multi-layer image guardrail for `text2image`, `img2img stylization`, and `img2img editing`,
plus `output` checks on generated results.

## What is implemented

- unified moderation API for input and output checks
- **text guard** (`text_guard`): RU/EN ML toxicity model (`cointegrated/rubert-tiny-toxicity` by default) fused with a stem lexicon; runs on both the prompt and OCR text
- **OCR adapter** for text baked into images
- **visual classifier** (zero-shot CLIP) — calibrated against a "safe" anchor so raw softmax noise no longer drives the verdict
- **explicit-content detector** (NSFW specialist)
- **LlavaGuard** (`llava_guard`): always-on trained policy-aware VLM (LlavaGuard-0.5B by default) — a second independent visual sensor alongside CLIP; verdicts mapped into the taxonomy
- **policy judge** = calibrated **fusion** (weighted noisy-OR) of all sensors + a **ShieldGemma escalation point** for borderline cases
- rule-based **decision engine** with `allow / review / block`
- a **web UI demo**, an HTTP API, and a CLI

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and [PLAN.md](PLAN.md) for what's next.

## Architecture

```text
request (prompt and/or image)
  -> text guard            (prompt: lexicon + ML toxicity)
  -> OCR                   (text inside the image)
       -> text guard       (OCR text)
  -> visual classifier     (zero-shot, calibrated vs safe anchor)
  -> explicit detector     (NSFW specialist)
  -> policy judge          (fusion = weighted noisy-OR; ShieldGemma escalation if borderline)
  -> decision engine       (thresholds, hard/soft rules)
  -> verdict + categories + confidence + evidence + per-signal breakdown
```

Each sensor only emits per-category scores in a uniform `SignalResult`. The fusion
collapses them into one calibrated score per category; the decision engine alone
decides `allow / review / block`.

## Quick start

1. Create and activate a virtual environment with Python 3.11+:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
```

2. Install the base dependencies:

```bash
pip install -r requirements.txt
```

3. For the ML sensors (visual classifier, NSFW detector, text toxicity model, LlavaGuard) install the extra ML deps:

```bash
pip install -r requirements-ml.txt
```

This installs **CPU** wheels by default. For an NVIDIA GPU, install torch + torchvision from the CUDA index first, then the ML file:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-ml.txt
```

Without these the service still runs — the ML sensors just report `skipped` (graceful degradation).

> **First run downloads model weights** into `.cache/huggingface` (LlavaGuard-0.5B is ~2 GB).
> On CPU LlavaGuard adds ~10–30 s per image; set `CENSOR_ENABLE_LLAVA_GUARD=false` to skip it
> (and its download) if you don't need the trained visual judge.

4. Install the Tesseract binary for your OS (needed only for OCR):

- Windows: install Tesseract so `tesseract.exe` is available, e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

The repo ships `tools/tessdata/eng.traineddata` and `tools/tessdata/rus.traineddata`, so Russian OCR does not depend on system language packs.

## JSONL logging demo

Run one command to execute a real moderation request, write JSONL logs, read them
back, and save a proof report:

```bash
python scripts/run_logging_demo.py
```

The runner writes `docs/logging_demo_run.md` with every command it executed, the
moderation response, row counts for the three JSONL streams, and a sample business
audit event.

Runtime logs are written by the production `JsonlLogSink`:

- `logs/logging_demo/system.jsonl`
- `logs/logging_demo/business_audit.jsonl`
- `logs/logging_demo/raw_payloads.jsonl`

Concurrent writers are serialized with a per-log-directory `.jsonl.lock` file, so
parallel threads and worker processes do not interleave JSONL lines.

This path uses no database, no external service runtime, no mock sink, and no
hidden fallback storage.

## Web UI demo (recommended way to see it)

A single self-contained file ([ui_demo.py](ui_demo.py)) — drop an image and/or type a prompt, get the verdict.

```bash
uvicorn ui_demo:app --reload
```

Open `http://127.0.0.1:8000`.

**How to use it:** type a prompt (optional), pick `scenario` / `stage`, drag-drop an
image (optional), press **Проверить**.

**What you see in the result panel:**
- **verdict** badge — `allow` (green) / `review` (amber) / `block` (red);
- **confidence** bar — the max fused score among the categories that fired;
- **categories** that fired, each with the sensors that raised it (evidence);
- **Сигналы сенсоров** — every sensor, its status (`ok/skipped/error`) and per-category scores;
- **Сведение (policy_fusion)** — the fusion breakdown: each category's fused score, the
  per-sensor contribution (`p`, weight), the agreement count, and the escalation status;
- raw JSON of the full response.

**What happens under the hood on submit** (`GuardrailPipeline.moderate`):
1. the **prompt** goes through the text guard (ML toxicity + lexicon);
2. **OCR** extracts text from the image; that text goes through the text guard again (`ocr_text_guard`);
3. the **visual classifier** (calibrated CLIP), the **NSFW detector**, and **LlavaGuard** (trained VLM judge) score the image;
4. the **policy judge** fuses all sensor scores per category via weighted noisy-OR; if any
   category lands in the gray zone `[review, block)` or it's an `output` stage with risk,
   it tries to **escalate to ShieldGemma** (currently a `skipped` stub until inference is wired);
5. the **decision engine** applies thresholds: a hard category blocks at `≥ block_threshold`;
   a soft category blocks only with agreement of ≥2 sensors or a ShieldGemma confirmation;
   anything `≥ review_threshold` but not blocked goes to `review`; otherwise `allow`.

## HTTP API

```bash
uvicorn censor_guard.app:app --reload      # API only
# docs (Swagger): http://127.0.0.1:8000/docs
```

`POST /v1/moderate` (the UI demo exposes the same logic at `POST /api/moderate`):

```jsonc
{
  "scenario": "output",
  "stage": "output",
  "prompt": "optional text",
  "image_path": "C:\\path\\to\\image.png"   // or "image_base64": "<...>"
}
```

## CLI example

```bash
python -m censor_guard.cli --scenario output --image-path path/to/image.png
```

## Бенчмарк и метрики

Бенчмарк-харнесс гоняет реальный image-классификатор по курируемому датасету
[`vekshinkir/image-censorship-small`](https://huggingface.co/datasets/vekshinkir/image-censorship-small)
(split `benchmarking`, 690 размеченных картинок) и считает метрики по всему пулу,
по каждой категории таксономии, по источникам и по AI-vs-реальным.

```bash
pip install -r requirements-benchmark.txt
python -m censor_guard.benchmark              # весь split benchmarking
python -m censor_guard.benchmark --limit 60   # быстрая проба
```

Что делает:
- прогресс-бар классификации с ETA (загрузка ~мгновенна из кэша HF);
- метрики: accuracy / precision / recall / F1 / FPR / ROC-AUC / PR-AUC, отдельно
  по категориям (one-vs-rest), по источникам данных и по срезу AI-генерация vs
  реальные; блок латентности (warmup, p95/p99, throughput);
- **отдельная секция устойчивости**: recall на adversarial-обфускациях
  (текст/шум/поворот/текстура) и FPR на edge-case hard-negatives — не размывают
  headline-метрики;
- **интерактивные графики Plotly** (матрица ошибок, распределение score, ROC по
  категориям, метрики по категориям, срезы по источникам и AI, латентность);
- подробный текстовый отчёт в консоль с вербальной оценкой по каждой категории;
- при `save_report` — папка `reports/benchmark_<timestamp>/` с самодостаточным
  `dashboard.html` (plotly.js встроен инлайн, открывается без сервера),
  `report.md`, `predictions.csv`, `metrics.json`.

Полезные флаги: `--limit N` (ограничить число картинок), `--split`,
`--no-save-report`, `--no-html`, `--output-dir`, `--seed`.

Программно (например, из ноутбука `notebooks/metrics_benchmark.ipynb`):

```python
from censor_guard.benchmark import run_benchmark, show_figures
result = run_benchmark(split="benchmarking", save_report=True)
show_figures(result["figures"])  # интерактивные Plotly-графики inline
```

## Environment variables

- `CENSOR_ENABLE_OCR=true|false`
- `CENSOR_ENABLE_VISUAL_CLASSIFIER=true|false`
- `CENSOR_ENABLE_EXPLICIT_DETECTOR=true|false`
- `CENSOR_ENABLE_TEXT_GUARD=true|false`
- `CENSOR_ENABLE_LLAVA_GUARD=true|false` — always-on trained visual judge (~10–30 s/image on CPU)
- `CENSOR_ENABLE_POLICY_JUDGE=true|false` — enables ShieldGemma escalation (fusion runs regardless)
- `CENSOR_VISUAL_MODEL_ID=openai/clip-vit-base-patch32`
- `CENSOR_EXPLICIT_MODEL_ID=Falconsai/nsfw_image_detection`
- `CENSOR_TEXT_MODEL_ID=cointegrated/rubert-tiny-toxicity` — empty string = lexicon only
- `CENSOR_LLAVA_GUARD_MODEL_ID=AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf`
- `CENSOR_LLAVA_GUARD_MAX_NEW_TOKENS=24` · `CENSOR_LLAVA_GUARD_UNSAFE_SCORE=0.9`
- `CENSOR_POLICY_JUDGE_MODEL_ID=google/shieldgemma-2-4b-it`
- `CENSOR_BLOCK_THRESHOLD=0.85`
- `CENSOR_REVIEW_THRESHOLD=0.55`
- `CENSOR_LOG_DIR=logs`
- `CENSOR_CALIBRATION_FLOOR=0.35` — CLIP calibration: lower = more sensitive, higher = more conservative
- `CENSOR_HF_CACHE_DIR=...`
- `CENSOR_TESSERACT_CMD=...`

OCR lookup order: project-local `tools/tessdata` → `CENSOR_TESSERACT_CMD` → `tesseract` on `PATH`
→ OS defaults (`C:\Program Files\Tesseract-OCR`, `/opt/homebrew/bin/tesseract`, `/usr/bin/tesseract`).
OCR tries `rus`, `eng`, and `rus+eng`.

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

- All ML sensors are optional. If a backend is missing, that sensor returns a structured
  `skipped` status instead of failing the request.
- The text guard's ML layer covers harassment/threat well but not every category; `sexual`
  in particular leans on the lexicon. The visual side of several categories (violence, self-harm,
  hate, etc.) currently relies on the weak CLIP-base signal — strengthening that is on the roadmap
  (more specialists / ShieldGemma). See [PLAN.md](PLAN.md).
- ShieldGemma is an escalation layer, not a CLIP analog. Its inference is not wired yet (the
  integration point returns `skipped`); until then the verdict rests on calibrated sensor fusion.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
