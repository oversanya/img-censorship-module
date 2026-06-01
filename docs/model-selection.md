# Model Selection

The baseline is optimized for a MacBook M4. It avoids 7B+ VLMs as default
dependencies and uses lazy loading so disabled detectors are never loaded.

## Default Models

### `AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf`

Primary visual safety judge. It is the smallest LlavaGuard v1.2 HF model and is
based on LLaVA-OneVision Qwen2 0.5B. It provides a structured safety rating,
category, and rationale. It should run on Apple Silicon with MPS, although it is
still the slowest default detector.

### `Falconsai/nsfw_image_detection`

Fast binary/multiclass NSFW detector. It is not enough on its own, but it is a
cheap prefilter for explicit sexual imagery.

### `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`

Small multilingual NLI model for prompt zero-shot classification. It helps with
Russian/English prompt variants without requiring a large LLM.

## Optional Models

### `google/shieldgemma-2-4b-it`

Stronger image safety model for sexual, dangerous, and violence policies. It is
gated by Google Gemma license and heavier than the default stack, so it is
disabled in `configs/pipeline.yaml`.

### `openai/clip-vit-base-patch32`

Optional zero-shot visual classifier for object/context hints. It is useful for
experiments but should not be treated as a final safety verdict.

## MacBook M4 Profile

Recommended order for local demos:

1. Run `--mock` to validate API shape.
2. Enable `prompt_keywords`, `prompt_zero_shot`, and `nsfw_vit`.
3. Enable LlavaGuard for final image checks.
4. Keep ShieldGemma disabled unless the machine has enough RAM and HF gated
   access is configured.

