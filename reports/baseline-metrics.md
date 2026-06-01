# Baseline Metrics Report

This report is generated from `examples/eval_manifest.example.csv` using the
Mac-friendly `configs/local.yaml` profile. It is a smoke benchmark for the demo,
not a final regulatory validation dataset.

## Dataset

- Total examples: 7
- Benign prompts: 2
- Unsafe prompts: 5
- Attack types covered: explicit prompt, case variation, spacing obfuscation,
  bank data

## Expected Local Result

```json
{
  "total": 7.0,
  "true_positive": 5.0,
  "false_positive": 0.0,
  "true_negative": 2.0,
  "false_negative": 0.0,
  "manual_review": 0.0,
  "precision_block": 1.0,
  "recall_block": 1.0,
  "f1_block": 1.0,
  "false_positive_rate": 0.0,
  "manual_review_rate": 0.0,
  "category_recall": {
    "dangerous": 1.0,
    "hate_extremism": 1.0,
    "personal_financial_data": 1.0
  },
  "attack_recall": {
    "bank_data": 1.0,
    "case_variation": 1.0,
    "explicit_prompt": 1.0,
    "spacing_obfuscation": 1.0
  }
}
```

## Interpretation

The local profile catches the included prompt-level high-severity cases and
does not block benign prompts. This does not prove image safety. Final scoring
must add safe and unsafe images, img2img laundering attempts, OCR cases, and
perturbed versions of unsafe images.
