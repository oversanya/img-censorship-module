# Threat Model

## Assets

- Bank brand and public distribution channels.
- User-uploaded images and generated outputs.
- User personal and financial data.
- Safety policy and audit logs.
- Model registry and detector weights.

## Adversaries

- Good-faith user receiving unsafe generated output by accident.
- Curious user probing the boundaries.
- Active attacker attempting to bypass the guardrail.
- Supply-chain attacker replacing model files, prompts, or thresholds.

## Attack Classes

| Attack | Example | Control |
| --- | --- | --- |
| Prompt obfuscation | slang, typos, mixed Russian/English, euphemisms | keyword guard plus multilingual NLI |
| Jailbreak prompt | "ignore the safety checker" | prompt guard and independent output guard |
| Img2img laundering | upload unsafe image and request "make it watercolor" | input image guard before generator |
| Output surprise | harmless prompt creates unsafe image | mandatory output image guard |
| Adversarial image transform | crop, blur, JPEG, noise, rotation | robustness test suite |
| Composition bypass | each artifact looks safe alone, together unsafe | prompt + image + output aggregation |
| Detector blind spot | one model misses niche content | ensemble and review path |
| Supply-chain tampering | unsafe model checkpoint or config change | pinned model IDs, hashes in audit log, config review |

## Operational Controls

- Run censor as a separate service, not inside the generator.
- All generation requests must pass through the censor gateway.
- Log detector model IDs, versions, thresholds, scores, and rationale.
- Fail closed for high-severity categories.
- Keep a red-team regression set with adversarial prompts and perturbed images.

