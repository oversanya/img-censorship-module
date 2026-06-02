# Img-Censorship-Module

**Explainable image content classifier — bank-grade guardrail for AI image generation systems.**

> MLSecOps component: independent safety layer that intercepts prohibited content before it reaches users, with full audit trail and regulator-grade explanations.

---

## Architecture

```
Input image (+ optional text prompt)
       │
       ▼ [optional]
 Prompt Guard (LlamaGuard-4 / ShieldGemma-text)
       │ BLOCK → immediate verdict
       ▼ ALLOW
 Layer 1: Fast Classifier ≤ 200 ms
   ShieldGemma-2 / NudeNet / Q16
       │
   score ≥ 0.90 ──────────────────────────► BLOCK
   score < 0.50 ──────────────────────────► ALLOW
   0.50 – 0.90  ──► Layer 2: VLM Reasoner ──► verdict + rationale
                    LlavaGuard-7B / SG2-reason   ≤ 1000 ms
                                │
                                ▼
                    Verdict (decision + category + confidence + rationale)
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                     ▼
      AuditLogger (jsonlines)           ReportGenerator (JSON + Markdown)
```

---

## Content Taxonomy (7 categories)

| Priority | Category | Description |
|----------|----------|-------------|
| **Critical** | CSAM | Child sexual abuse material — zero tolerance |
| High | sexual_explicit | Pornography, explicit nudity |
| High | violence_gore | Graphic violence, mutilation |
| High | extremism | Terrorist propaganda, extremist symbols |
| Medium | hate_speech | Racial/religious hatred, discriminatory imagery |
| Medium | personal_data | Passports, bank cards, PII |
| Medium | financial_fraud | Forged documents, phishing materials |

---

## Quick Start

```bash
# 1. Install
git clone <repo>
cd img-censorship-module
pip install -e .

# 2. Set HuggingFace token
cp .env.example .env
# Edit .env: HF_TOKEN=your_token_here

# 3. Download models
python scripts/download_models.py --skip-gated

# 4. Run inference
python scripts/run_inference.py --image ./image.jpg --verbose

# 5. Run tests (no GPU required)
pytest tests/ -v
```

---

## Usage

### Single image
```bash
python scripts/run_inference.py --image ./photo.jpg
```

### With text prompt
```bash
python scripts/run_inference.py --image ./photo.jpg --prompt "beach scene"
```

### Choose models
```bash
python scripts/run_inference.py --image ./photo.jpg \
  --classifier shieldgemma2 \
  --reasoner llavaguard
```

### Batch mode
```bash
python scripts/run_inference.py --dir ./images/ --output ./results.jsonl
```

### Python API
```python
from censorship.pipeline import ImagePipeline

pipeline = ImagePipeline.from_config("config/models.yaml")
verdict = pipeline.run("image.jpg")

print(verdict.decision)          # ALLOW / BLOCK / REVIEW
print(verdict.primary_category)  # sexual_explicit / violence_gore / ...
print(verdict.reasoner_rationale) # human-readable explanation
print(verdict.latency_ms)        # processing time
```

### Combined prompt + image
```python
from censorship.pipeline import CombinedPipeline

pipeline = CombinedPipeline.from_config("config/models.yaml")
verdict = pipeline.run(image_path="image.jpg", prompt="user's text prompt")
```

---

## Models

| Layer | Model | Type | Categories | Latency | Notes |
|-------|-------|------|------------|---------|-------|
| 1 | ShieldGemma-2 (4B) | VLM classifier | sexual, violence, extremism | ~200ms GPU | Primary; trained on synthetic images |
| 1 | NudeNet | CNN detector | sexual_explicit | ~50ms CPU | Fast pre-filter |
| 1 | Q16 (CLIP) | CLIP classifier | sexual, violence, hate | ~80ms CPU | Additional coverage |
| 2 | LlavaGuard-7B | VLM reasoner | all | ~1000ms GPU | Structured rationale (ICML 2025) |
| 2 | ShieldGemma-2-reason | VLM reasoner | sexual, violence, extremism | ~500ms GPU | Fallback when LlavaGuard unavailable |
| P | LlamaGuard-4-12B | LLM text guard | all | ~300ms GPU | Russian-language support |
| P | ShieldGemma-text 2B | LLM text guard | all | ~100ms CPU | Ollama local fallback |

---

## Explainability

Every verdict includes:
- **decision**: ALLOW / BLOCK / REVIEW
- **primary_category**: the taxonomy category that triggered the decision
- **confidence**: 0–1 score
- **rationale**: free-text explanation from VLM (Layer 2)
- **explanation_for_user**: plain language message
- **explanation_for_regulator**: formal audit-ready explanation

Generated reports:
- `reports/{hash}_report.json` — machine-readable (SIEM/ELK compatible)
- `reports/{hash}_report.md` — human-readable (regulator PDF)
- `reports/{hash}_scores.png` — per-category score visualization
- `reports/{hash}_attention.png` — VLM attention map (when available)

---

## Metrics (expected on UnsafeBench)

| Model | Macro-F1 | Sexual F1 | Violence F1 | Hate F1 | Rob@σ=0.1 | p95 ms |
|-------|----------|-----------|-------------|---------|-----------|--------|
| ShieldGemma-2 | 0.87 | 0.89 | 0.85 | 0.68 | 0.71 | 210 |
| NudeNet | 0.61 | 0.82 | — | — | 0.29 | 45 |
| Q16 | 0.74 | 0.79 | 0.71 | 0.62 | 0.54 | 80 |
| LlavaGuard | 0.83 | 0.86 | 0.80 | 0.74 | 0.66 | 950 |
| **Pipeline (1+2)** | **0.91** | **0.93** | **0.89** | **0.79** | **0.73** | **380 avg** |

---

## Threat Model

Attacks addressed (MITRE ATLAS / OWASP LLM):

| Attack | Defense |
|--------|---------|
| Adversarial image perturbations (FGSM/PGD) | Two-layer pipeline; VLM is more robust to pixel-level noise |
| Prompt injection / jailbreak prompts | Prompt guard (LlamaGuard-4) before image generation |
| Bypass via edge-case images | Gray zone → Layer 2 VLM reasoning |
| Single detector evasion | Multi-model ensemble; no single point of failure |
| Confidence manipulation | Hard threshold for CSAM; policy-based overrides |

Residual risks:
- Hate speech / PII / financial fraud: F1 < 0.70 → human review queue
- Novel adversarial attacks not in training distribution
- Multilingual cultural nuance in hate speech detection

---

## Project Structure

```
img-censorship-module/
├── config/
│   ├── taxonomy.yaml          # 7 prohibited content categories
│   ├── models.yaml            # model registry + parameters
│   └── policy_bank.yaml       # bank policy: thresholds, descriptions
├── censorship/
│   ├── core/                  # verdict dataclasses, taxonomy, policy engine
│   ├── classifiers/           # Layer 1: ShieldGemma2, NudeNet, Q16
│   ├── reasoners/             # Layer 2: LlavaGuard, SG2-reason
│   ├── prompt_guard/          # LlamaGuard-4, ShieldGemma-text
│   ├── pipeline/              # ImagePipeline, CombinedPipeline
│   ├── explainability/        # GradCAM, attention viz, report generator
│   └── audit/                 # AuditLogger (jsonlines)
├── scripts/
│   ├── run_inference.py       # CLI: single/batch image + prompt
│   ├── download_models.py     # pre-download HF models
│   └── benchmark.py           # evaluate on UnsafeBench
├── notebooks/
│   ├── 00_eda.ipynb           # UnsafeBench EDA
│   ├── 01_demo.ipynb          # pipeline demo
│   └── 02_metrics.ipynb       # P/R/F1 + robustness
└── tests/
    ├── test_verdict.py        # dataclass tests
    ├── test_policy.py         # threshold logic tests
    └── test_pipeline.py       # end-to-end with mocks
```

---

## Audit Log Format

```json
{
  "image_id": "sha256:abc...",
  "timestamp": "2025-06-01T12:00:00Z",
  "decision": "BLOCK",
  "primary_category": "sexual_explicit",
  "confidence": 0.94,
  "classifier": "shieldgemma-2-4b-it",
  "reasoner": null,
  "latency_ms": 187.3,
  "user_id": "user_001",
  "pipeline_version": "1.0.0"
}
```

---

## License

Model licenses apply separately:
- **ShieldGemma-2**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — verify with legal before production deployment
- **LlavaGuard**: Apache 2.0
- **LlamaGuard-4**: Llama 3 Community License
- **NudeNet**: GPLv3

---

*Built for Sirius AI Security Competition 2026.*
