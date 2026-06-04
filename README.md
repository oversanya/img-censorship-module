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

3. Run the API:

```bash
uvicorn censor_guard.app:app --reload
```

4. Open the docs:

`http://127.0.0.1:8000/docs`

## CLI example

```bash
python -m censor_guard.cli --scenario output --image-path path/to/image.png
```

## Бенчмарк и метрики

Полный бенчмарк-харнесс гоняет реальный image-классификатор по датасетам из
[`censor_benchmark_datasets.md`](censor_benchmark_datasets.md) и считает метрики
по всему пулу, по каждой категории таксономии и по каждому датасету.

```bash
pip install -r requirements-benchmark.txt
# gated-датасеты (🔒) подтянутся при наличии HF_TOKEN:
export HF_TOKEN=hf_...
python -m censor_guard.benchmark --n-dataset 100 --save-report
```

Что делает:
- двухфазный прогресс с ETA (загрузка датасетов → классификация);
- метрики: accuracy / precision / recall / F1 / FPR / ROC-AUC / PR-AUC, отдельно
  по категориям (one-vs-rest) и по датасетам; блок латентности (warmup, p95/p99,
  throughput); отдельный учёт adversarial / hard-negative;
- графики (confusion matrix, распределение score, ROC, метрики по категориям и
  датасетам, латентность);
- подробный текстовый отчёт в консоль с вербальной оценкой по каждой категории;
- при `--save-report` — папка `reports/benchmark_<timestamp>/` с `report.md`,
  `dashboard.html`, `predictions.csv`, `metrics.json` и `figures/*.png`.

Полезные флаги: `--n-dataset N` (картинок на датасет), `--datasets a,b,c`
(подмножество по ключам датасетов или кодам категорий), `--no-save-report`,
`--no-html`, `--output-dir`, `--seed`, `--max-load-seconds` (лимит на один
датасет, по умолчанию 120с), `--max-total-load-seconds` (общий лимит фазы
загрузки, по умолчанию 300с — дальше оставшиеся датасеты пропускаются, чтобы
прогон не «висел» на медленных/недоступных источниках).

Недоступные датасеты (gated без принятых условий, переехавшие/script-based, или
зеркала без байтов картинок) автоматически пропускаются с понятной причиной —
бенчмарк продолжает работу на доступных. Каждый источник грузится в отдельном
потоке с жёстким таймаутом, поэтому зависшая сеть не блокирует прогон.

Программно (например, из ноутбука):

```python
from censor_guard.benchmark import run_benchmark
result = run_benchmark(n_dataset=100, save_report=True)
result["figures"]["overview"]  # matplotlib-фигуры для inline-показа
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
- `CENSOR_TESSERACT_CMD`, if set
- `tesseract` from `PATH`
- Windows defaults under `C:\Program Files\Tesseract-OCR`
- macOS defaults such as `/opt/homebrew/bin/tesseract` and `/usr/local/bin/tesseract`
- Linux default `/usr/bin/tesseract`

## Notes

- The text filter is a placeholder by design. It always returns `safe`, but the adapter contract is already in place.
- The OCR, visual classifier, explicit detector, and ShieldGemma judge are optional adapters. If a backend is unavailable, the service returns a structured `skipped` status instead of failing the whole request.
- The heuristic policy judge keeps the pipeline operational before a real multimodal judge is attached.
