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
