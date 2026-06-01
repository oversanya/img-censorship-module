# Unresolved Risks

This prototype is a baseline guardrail, not a certified production control.

## Known Gaps

- Keyword rules miss novel euphemisms and culturally specific coded language.
- The default Mac profile disables LlavaGuard and ShieldGemma to avoid heavy
  downloads; image reasoning is therefore limited until those layers are
  enabled.
- OCR is not implemented yet, so text embedded inside images can be missed.
- Image perturbation testing is represented in the methodology but not automated
  in this repo yet.
- Thresholds are not calibrated on a bank-owned validation set.
- The prototype logs audit fields in the response, but does not yet write to
  immutable storage.
- Supply-chain controls such as model hash pinning and signed configs are
  documented as requirements but not enforced.

## Next Hardening Steps

1. Add OCR detector for text in images and screenshots.
2. Add image perturbation test generation: crop, blur, JPEG compression, noise,
   rotation, and contrast shifts.
3. Calibrate thresholds on a labeled bank validation set.
4. Enable LlavaGuard for output images by default on machines with enough RAM.
5. Store audit events in append-only logs with model/config version hashes.
6. Add human review queue integration for `review` verdicts.

