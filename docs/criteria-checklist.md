# Case Criteria Checklist

| Requirement | Artifact in repo |
| --- | --- |
| Define prohibited categories with rationale | `docs/taxonomy.md`, `configs/policy.yaml`, `src/img_censor/policy.py` category metadata |
| Review open detection approaches | `docs/model-review.md`, `docs/model-selection.md` |
| Build working baseline prototype | CLI `img-censor`, FastAPI `/v1/censor`, `src/img_censor/pipeline.py` |
| Return verdict, category, and rationale | `GuardResult` JSON from CLI/API |
| Evaluate with measurable method | `docs/evaluation-methodology.md`, `scripts/evaluate_manifest.py`, `reports/baseline-metrics.md` |
| Threat model for active attacker | `docs/threat-model.md` |
| Residual risks | `docs/unresolved-risks.md` |
| Demo materials | `docs/demo-script.md`, README quick start |
| Independent control from generator | `docs/architecture.md`, API service design |
| Defense in depth, not one detector | `configs/pipeline.yaml`, model stack docs |
| Regulator explanation | audit fields: policy version, detector, category, score, rationale, category evidence, stage, latency |
| Manual review workflow | `outputs/review_queue.jsonl` configured by `decision.review_queue_path` |

## Current Demo Scope

The local profile is intentionally lightweight and can run on a MacBook. It
demonstrates the integration pattern, prompt guard, fast image detector wiring,
audit response shape, and metrics workflow.

The full image-safety story requires enabling image VLM layers and evaluating on
labeled safe/unsafe images before claiming production-level coverage.
