# Open Model Review

This baseline uses a defense-in-depth stack rather than a single safety model.
The goal is to make bypass harder and keep a regulator-readable audit trail.

| Model or approach | Used as | Strengths | Weaknesses |
| --- | --- | --- | --- |
| `Falconsai/nsfw_image_detection` | Fast image detector | Small, cheap, easy to run locally, good first pass for explicit imagery | Narrow taxonomy; does not reason about context, hate symbols, banking data, or composite violations |
| `AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf` | Optional VLM image safety judge | Lightweight LlavaGuard variant; can reason over image context and produce a category/rationale | Slower than small classifiers; generative output parsing can be brittle; still needs policy tuning |
| `google/shieldgemma-2-4b-it` | Optional stronger image safety classifier | Google model for image safety classification across sexual, dangerous, and violence/gore policies; outputs Yes/No probabilities | Gated license, 4B size, narrower native policy set than bank taxonomy |
| `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | Optional prompt zero-shot detector | Lightweight multilingual text classification for Russian/English prompt risk | Text-only; weaker than dedicated safety LLMs; can miss adversarial euphemisms |
| Keyword and normalization guard | Default prompt prefilter | Transparent, fast, auditable, catches high-severity known bad prompts before model download | High maintenance; cannot cover semantic novelty alone |
| CLIP zero-shot | Optional visual hints | Lightweight broad visual semantics and better robustness than some small custom classifiers | Not a safety model; scores need calibration |

## Source Notes

- ShieldGemma 2 is described by Google as a 4B image safety classifier for
  synthetic and natural images, with native categories for sexually explicit,
  dangerous, and violence/gore content.
- UnsafeBench is useful for evaluation design because it includes human-labeled
  real and AI-generated unsafe images and explicitly evaluates robustness under
  perturbed images.
- LlavaGuard is included as a VLM-based safety layer because a VLM can catch
  contextual and compositional image risks that narrow classifiers miss.
- Haystack's guardrail cookbook motivates routing/guardrail integration rather
  than treating moderation as an offline classifier only.

## Baseline Decision

The Mac-friendly default profile enables only:

- prompt keyword guard;
- fast NSFW image detector.

The fuller profile enables optional prompt NLI, LlavaGuard, CLIP, and
ShieldGemma depending on available memory, latency budget, and license access.

