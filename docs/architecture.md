# Architecture

## Goal

Build an independent image censor module for a bank-owned GenAI image system.
The module is not a classifier demo. It is a security control that must work
against accidental unsafe outputs and active bypass attempts.

## Pipeline

1. `prompt guard`
   - Checks text before generation.
   - Catches explicit intent, jailbreak-like phrasing, obfuscation, and risky
     banking data requests.
   - Uses keyword heuristics plus a lightweight multilingual NLI classifier.

2. `input image guard`
   - Checks uploaded images before img2img restyling or editing.
   - Prevents laundering unsafe content through a "style transfer" operation.
   - Uses a fast NSFW classifier and LlavaGuard 0.5B.

3. `generation service`
   - Treated as untrusted from the guardrail perspective.
   - The generator cannot approve its own input or output.

4. `output image guard`
   - Checks the final generated image.
   - This is mandatory because harmless prompts can still produce unsafe images.
   - Uses the same image detector ensemble as the input guard.

5. `decision aggregator`
   - Normalizes findings into `allow`, `review`, or `block`.
   - Applies high-severity fail-closed categories.
   - Emits audit data: detector versions, scores, rationale, thresholds, and
     model IDs.

## Default Lightweight Model Stack

| Layer | Model | Size profile | Role |
| --- | --- | --- | --- |
| Prompt zero-shot | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | about 0.1B | Multilingual prompt risk classification |
| Fast NSFW image | `Falconsai/nsfw_image_detection` | about 86M | Cheap explicit-content prefilter |
| VLM safety judge | `AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf` | about 0.9B params, 1.8GB weights | Policy-aware visual moderation with rationale |
| Optional stronger judge | `google/shieldgemma-2-4b-it` | 4B, gated | Stronger image safety classifier for sexual, dangerous, violence |
| Optional broad visual semantics | `openai/clip-vit-base-patch32` | about 151M | Zero-shot object/context hints |

## Why Defense in Depth

One detector is brittle. The baseline uses several imperfect detectors because
they fail differently:

- keyword guard is cheap and explainable;
- MiniLM NLI catches prompt variants and Russian/English paraphrases;
- NSFW ViT is fast and specialized;
- LlavaGuard reasons over full visual context and can return a rationale;
- ShieldGemma 2 can be enabled for a stronger second opinion where its gated
  license is acceptable.

## Decision Semantics

- `allow`: no detector reaches review threshold.
- `review`: medium-confidence finding, disagreement between detectors, or a
  category that is risky but context-dependent.
- `block`: high-confidence violation or fail-closed category.

For banking, false negatives are more expensive for high-severity categories,
so the default policy biases toward blocking sexual exploitation, extremist
content, self-harm instructions, and personal or financial data leakage.

