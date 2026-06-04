# Censor Module MVP

Multi-layer image guardrail for `text2image`, `img2img stylization`, and `img2img editing`,
plus `output` checks on generated results.

## What is implemented

- unified moderation API for input and output checks
- **text guard** (`text_guard_heuristic`): RU/EN ML toxicity model (`cointegrated/rubert-tiny-toxicity` by default) fused with a stem lexicon; runs on both the prompt and OCR text
- **OCR adapter** for text baked into images
- **visual classifier** (zero-shot CLIP) — calibrated against a "safe" anchor so raw softmax noise no longer drives the verdict
- **explicit-content detector** (NSFW specialist)
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

1. Create an environment with Python 3.11+.
2. Install the base dependencies:

```bash
pip install -r requirements.txt
```

3. For the ML sensors (visual classifier, NSFW detector, text toxicity model) install the extra ML deps:

```bash
pip install -r requirements-ml.txt
```

Without these the service still runs — the ML sensors just report `skipped` (graceful degradation).

4. Install the Tesseract binary for your OS (needed only for OCR):

- Windows: install Tesseract so `tesseract.exe` is available, e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

The repo ships `tools/tessdata/eng.traineddata` and `tools/tessdata/rus.traineddata`, so Russian OCR does not depend on system language packs.

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
2. **OCR** extracts text from the image; that text goes through the text guard again (`ocr_text_guard_heuristic`);
3. the **visual classifier** (calibrated CLIP) and the **NSFW detector** score the image;
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

## Environment variables

- `CENSOR_ENABLE_OCR=true|false`
- `CENSOR_ENABLE_VISUAL_CLASSIFIER=true|false`
- `CENSOR_ENABLE_EXPLICIT_DETECTOR=true|false`
- `CENSOR_ENABLE_TEXT_GUARD=true|false`
- `CENSOR_ENABLE_POLICY_JUDGE=true|false` — enables ShieldGemma escalation (fusion runs regardless)
- `CENSOR_VISUAL_MODEL_ID=openai/clip-vit-base-patch32`
- `CENSOR_EXPLICIT_MODEL_ID=Falconsai/nsfw_image_detection`
- `CENSOR_TEXT_MODEL_ID=cointegrated/rubert-tiny-toxicity` — empty string = lexicon only
- `CENSOR_POLICY_JUDGE_MODEL_ID=google/shieldgemma-2-4b-it`
- `CENSOR_BLOCK_THRESHOLD=0.85`
- `CENSOR_REVIEW_THRESHOLD=0.55`
- `CENSOR_CALIBRATION_FLOOR=0.35` — CLIP calibration: lower = more sensitive, higher = more conservative
- `CENSOR_HF_CACHE_DIR=...`
- `CENSOR_TESSERACT_CMD=...`

OCR lookup order: project-local `tools/tessdata` → `CENSOR_TESSERACT_CMD` → `tesseract` on `PATH`
→ OS defaults (`C:\Program Files\Tesseract-OCR`, `/opt/homebrew/bin/tesseract`, `/usr/bin/tesseract`).
OCR tries `rus`, `eng`, and `rus+eng`.

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
