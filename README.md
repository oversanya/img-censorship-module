# Image Censorship Module

MLSecOps guardrail module for image generation pipelines. The module checks
text prompts, input images for img2img flows, and final generated images. It
returns a machine-readable verdict, violated category, detector evidence, and a
human-readable rationale.

The default local profile is intentionally lightweight enough for a MacBook:

- `Falconsai/nsfw_image_detection` as a fast NSFW image classifier.
- a local keyword prompt guard for obvious high-risk prompt requests.
- `cointegrated/rubert-tiny-toxicity` as a tiny local Russian prompt toxicity classifier.
- local OCR and QR-code detectors for text/links embedded inside images.
- optional `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` as a multilingual zero-shot prompt classifier.
- optional `AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf` for stronger image reasoning.
- optional `google/shieldgemma-2-4b-it` for stronger gated image safety checks.

## Architecture

```mermaid
flowchart LR
    A["User prompt"] --> B["Prompt guard"]
    C["Input image for img2img"] --> D["Input image guard"]
    B --> E["Generation service"]
    D --> E
    E --> F["Output image guard"]
    F --> G["Decision aggregator"]
    B --> G
    D --> G
    G --> H["allow / review / block + audit log"]
```

The censor module must run as an independent service in front of and behind the
generator. The generator is not trusted to make policy decisions about its own
outputs.

## Hackathon Flow

The demo implementation exposes the three required censor stages separately and
also provides a full guarded flow with a mock generator:

```mermaid
flowchart TD
    P["Prompt"] --> PC["/v1/censor/prompt<br/>Prompt Censor"]
    I["Input image for img2img"] --> IC["/v1/censor/input-image<br/>Input Image Censor"]
    PC --> AG["Input Gate Aggregator"]
    IC --> AG
    AG -->|block/review| STOP["Stop before generation"]
    AG -->|allow| GEN["Generator<br/>mock in hackathon mode"]
    GEN --> OC["/v1/censor/output-image<br/>Output Image Censor"]
    OC --> FINAL["Final allow/review/block"]
    FINAL --> AUDIT["Audit trail"]
```

Run the full local demo flow:

```bash
scripts/run_hackathon_flow.sh "Сгенерируй фото машины"
```

Run only one stage:

```bash
.venv/bin/img-censor --config configs/local.yaml --stage prompt --prompt "Нарисуй свастику"
.venv/bin/img-censor --config configs/local.yaml --stage input-image --input-image ./samples/input.png
.venv/bin/img-censor --config configs/local.yaml --stage output-image --output-image ./samples/generated.png
```

## Quick Start

Install everything into a project-local virtual environment:

```bash
scripts/install_local.sh
```

OCR text-in-image checks use `pytesseract` when the local Tesseract binary is
available. On macOS, install it with `brew install tesseract tesseract-lang`.
QR decoding runs locally through OpenCV.

Start the local API:

```bash
scripts/run_local_api.sh
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

Run a prompt check from the terminal:

```bash
curl -X POST http://127.0.0.1:8000/v1/censor \
  -F 'prompt=make a realistic promo image for a bank card'
```

Or call separate hackathon endpoints:

```bash
curl -X POST http://127.0.0.1:8000/v1/censor/prompt \
  -F 'prompt=Нарисуй свастику'

curl -X POST http://127.0.0.1:8000/v1/censor/full \
  -F 'prompt=Сгенерируй фото машины'
```

Run an image check from the terminal:

```bash
curl -X POST http://127.0.0.1:8000/v1/censor \
  -F 'output_image=@./samples/generated.png'
```

Dry-run the CLI without downloading models:

```bash
.venv/bin/img-censor --config configs/local.yaml --prompt "safe banking banner" --mock
```

Type prompts directly in the terminal:

```bash
scripts/censor_prompt.sh
```

Or pass one prompt as script arguments:

```bash
scripts/censor_prompt.sh Нарисуй свастику
```

Pre-download enabled local models:

```bash
.venv/bin/python scripts/download_models.py --config configs/local.yaml
```

Pre-download only the local prompt classification model:

```bash
scripts/download_prompt_model.sh
```

Use the fuller local model profile:

```bash
IMG_CENSOR_CONFIG=configs/pipeline.yaml scripts/run_local_api.sh
```

Use a stricter CLI profile by lowering the block threshold:

```bash
.venv/bin/img-censor --config configs/local.yaml --output-image ./image.png --block-threshold 0.55
```

Evaluate a CSV manifest:

```bash
.venv/bin/python scripts/evaluate_manifest.py examples/eval_manifest.example.csv --config configs/local.yaml
```

## Project Layout

```text
configs/local.yaml             Mac-friendly default runtime profile
configs/pipeline.yaml          Fuller runtime model registry and thresholds
configs/policy.yaml            Editable bank policy taxonomy and keyword rules
docs/architecture.md           End-to-end pipeline design
docs/taxonomy.md               Prohibited content taxonomy
docs/threat-model.md           MLSecOps threat model
docs/model-selection.md        Lightweight model choices for MacBook M4
docs/model-review.md           Open detector review and tradeoffs
docs/evaluation-methodology.md Metrics and benchmark plan
docs/unresolved-risks.md       Remaining production hardening risks
docs/demo-script.md            Demo flow for the defense
docs/hackathon-pipeline.md     Implemented prompt/input/output censor flow
docs/criteria-checklist.md     Mapping from case criteria to artifacts
reports/baseline-metrics.md    Baseline local metrics report
src/img_censor/                Pipeline implementation
src/img_censor/review_queue.py JSONL queue for manual review decisions
src/img_censor/__main__.py     Allows python -m img_censor CLI usage
scripts/install_local.sh       Create .venv and install local dependencies
scripts/run_local_api.sh       Start the local FastAPI service
scripts/censor_prompt.sh       Prompt-in-terminal censor runner
scripts/download_prompt_model.sh Download the local prompt classifier
scripts/run_hackathon_flow.sh  Full input-gate/mock-generator/output-gate demo
tests/                         Tests that do not download models
models/hf-cache/               Local Hugging Face cache, contents ignored
samples/                       Local demo images, contents ignored
outputs/review_queue.jsonl     Runtime manual-review queue, contents ignored
```

## Notes

The API runs fully on your machine. The first real image check may download an
enabled Hugging Face model into `models/hf-cache`; after that, inference uses
the local cache. ShieldGemma 2 is kept optional because the model is gated and
requires accepting the Google Gemma license.

Review decisions are appended to `outputs/review_queue.jsonl`. Every response
also includes scenario, request id, triggered stages, and detector latency in
the `audit` block.
