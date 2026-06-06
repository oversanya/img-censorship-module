# JSONL Logging Demo Run

- Started at: `2026-06-06T06:21:43.956143+00:00`
- Status: `PASS`
- Log dir: `C:\Users\hex\.codex\worktrees\d322\img-censorship-module\logs\logging_demo`
- Python: `3.12.13 (main, Mar  3 2026, 15:01:35) [MSC v.1944 64 bit (AMD64)]`

## Commands And Outputs

### 1. Check Python dependencies

Command:

```text
C:\Users\hex\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -c import pydantic, PIL; print('required Python dependencies are installed')
```

Exit code: `0`

Stdout:

```text
required Python dependencies are installed
```

Stderr:

```text
<empty>
```

### 2. Run JSONL logging verification

Command:

```text
C:\Users\hex\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/verify_logging_demo.py --log-dir C:\Users\hex\.codex\worktrees\d322\img-censorship-module\logs\logging_demo --clean
```

Exit code: `0`

Stdout:

```text
{"status":"PASS","log_dir":"C:\\Users\\hex\\.codex\\worktrees\\d322\\img-censorship-module\\logs\\logging_demo","request_id":"ceeef0a4-2d57-4857-bf2d-113f5b2e9d70","verdict":"allow","audit_reason_code":"no_policy_signal_above_review_threshold","audit_system_trace_id":"8b011928-fb92-48db-bb12-b489278da62c","counts":{"system_jsonl_rows":6,"business_audit_jsonl_rows":1,"raw_payloads_jsonl_rows":1},"checks":{"response_has_audit":true,"system_jsonl_exists":true,"business_audit_jsonl_exists":true,"raw_payloads_jsonl_exists":true,"system_jsonl_rows_gte_1":true,"business_audit_jsonl_rows_eq_1":true,"raw_payloads_jsonl_rows_eq_1":true},"audit_sample":{"audit_id":"0a0cabd1-3fd8-476d-a40b-7c5d71a51e95","trace_id":"8b011928-fb92-48db-bb12-b489278da62c","request_id":"ceeef0a4-2d57-4857-bf2d-113f5b2e9d70","scenario":"output","stage":"output","verdict":"allow","reason_code":"no_policy_signal_above_review_threshold","human_reason":"Allowed because no policy signal reached the review threshold.","categories":[],"confidence":0.0,"thresholds":{"block":0.85,"review":0.55},"evidence":{},"fusion_contributions":{},"signals_summary":[{"name":"text_guard","status":"ok","categories":{},"text":[],"reason":"No flagged terms in text."},{"name":"policy_fusion","status":"ok","categories":{},"text":[],"reason":"Fused calibrated sensor evidence via weighted noisy-OR."}],"raw_payload":{"request":{"request_id":"ceeef0a4-2d57-4857-bf2d-113f5b2e9d70","scenario":"output","stage":"output","prompt":"A harmless product photo of a ceramic mug on a desk.","image_path":null,"image_base64":null,"metadata":{}},"response":{"request_id":"ceeef0a4-2d57-4857-bf2d-113f5b2e9d70","scenario":"output","stage":"output","verdict":"allow","categories":[],"confidence":0.0,"reason":"Allowed because no policy signal reached the review threshold.","evidence":{},"audit":{"audit_id":"0a0cabd1-3fd8-476d-a40b-7c5d71a51e95","reason_code":"no_policy_signal_above_review_threshold","human_reason":"Allowed because no policy signal reached the review threshold.","policy_version":"logging-demo","thresholds":{"block":0.85,"review":0.55},"matched_categories":[],"decision_path":[{"step":"collect_scores","scores":{},"sources":{},"agreement":{}},{"step":"apply_block_rules","threshold":0.85,"hard_block_matches":[],"soft_block_matches":[]},{"step":"apply_review_rules","threshold":0.55,"review_matches":[]},{"step":"final_verdict","verdict":"allow","reason_code":"no_policy_signal_above_review_threshold"}],"system_trace_id":"8b011928-fb92-48db-bb12-b489278da62c"},"signals":[{"name":"text_guard","status":"ok","categories":{},"text":[],"reason":"No flagged terms in text.","raw":{"mode":"lexicon","lexicon_matches":{},"text_length":52}},{"name":"policy_fusion","status":"ok","categories":{},"text":[],"reason":"Fused calibrated sensor evidence via weighted noisy-OR.","raw":{"mode":"weighted_noisy_or","contributions":{},"agreement":{},"escalation":{"attempted":false,"needed":false}}}],"notes":[]},"signals":[{"name":"text_guard","status":"ok","categories":{},"text":[],"reason":"No flagged terms in text.","raw":{"mode":"lexicon","lexicon_matches":{},"text_length":52}},{"name":"policy_fusion","status":"ok","categories":{},"text":[],"reason":"Fused calibrated sensor evidence via weighted noisy-OR.","raw":{"mode":"weighted_noisy_or","contributions":{},"agreement":{},"escalation":{"attempted":false,"needed":false}}}],"notes":[]},"created_at":"2026-06-06T06:21:44.397603Z"},"moderation_response":{"request_id":"ceeef0a4-2d57-4857-bf2d-113f5b2e9d70","scenario":"output","stage":"output","verdict":"allow","categories":[],"confidence":0.0,"reason":"Allowed because no policy signal reached the review threshold.","evidence":{},"audit":{"audit_id":"0a0cabd1-3fd8-476d-a40b-7c5d71a51e95","reason_code":"no_policy_signal_above_review_threshold","human_reason":"Allowed because no policy signal reached the review threshold.","policy_version":"logging-demo","thresholds":{"block":0.85,"review":0.55},"matched_categories":[],"decision_path":[{"step":"collect_scores","scores":{},"sources":{},"agreement":{}},{"step":"apply_block_rules","threshold":0.85,"hard_block_matches":[],"soft_block_matches":[]},{"step":"apply_review_rules","threshold":0.55,"review_matches":[]},{"step":"final_verdict","verdict":"allow","reason_code":"no_policy_signal_above_review_threshold"}],"system_trace_id":"8b011928-fb92-48db-bb12-b489278da62c"},"signals":[{"name":"text_guard","status":"ok","categories":{},"text":[],"reason":"No flagged terms in text.","raw":{"mode":"lexicon","lexicon_matches":{},"text_length":52}},{"name":"policy_fusion","status":"ok","categories":{},"text":[],"reason":"Fused calibrated sensor evidence via weighted noisy-OR.","raw":{"mode":"weighted_noisy_or","contributions":{},"agreement":{},"escalation":{"attempted":false,"needed":false}}}],"notes":[]}}
```

Stderr:

```text
<empty>
```

## Verification Summary

```json
{
  "status": "PASS",
  "log_dir": "C:\\Users\\hex\\.codex\\worktrees\\d322\\img-censorship-module\\logs\\logging_demo",
  "request_id": "ceeef0a4-2d57-4857-bf2d-113f5b2e9d70",
  "verdict": "allow",
  "audit_reason_code": "no_policy_signal_above_review_threshold",
  "audit_system_trace_id": "8b011928-fb92-48db-bb12-b489278da62c",
  "counts": {
    "system_jsonl_rows": 6,
    "business_audit_jsonl_rows": 1,
    "raw_payloads_jsonl_rows": 1
  },
  "checks": {
    "response_has_audit": true,
    "system_jsonl_exists": true,
    "business_audit_jsonl_exists": true,
    "raw_payloads_jsonl_exists": true,
    "system_jsonl_rows_gte_1": true,
    "business_audit_jsonl_rows_eq_1": true,
    "raw_payloads_jsonl_rows_eq_1": true
  },
  "audit_sample": {
    "audit_id": "0a0cabd1-3fd8-476d-a40b-7c5d71a51e95",
    "trace_id": "8b011928-fb92-48db-bb12-b489278da62c",
    "request_id": "ceeef0a4-2d57-4857-bf2d-113f5b2e9d70",
    "scenario": "output",
    "stage": "output",
    "verdict": "allow",
    "reason_code": "no_policy_signal_above_review_threshold",
    "human_reason": "Allowed because no policy signal reached the review threshold.",
    "categories": [],
    "confidence": 0.0,
    "thresholds": {
      "block": 0.85,
      "review": 0.55
    },
    "evidence": {},
    "fusion_contributions": {},
    "signals_summary": [
      {
        "name": "text_guard",
        "status": "ok",
        "categories": {},
        "text": [],
        "reason": "No flagged terms in text."
      },
      {
        "name": "policy_fusion",
        "status": "ok",
        "categories": {},
        "text": [],
        "reason": "Fused calibrated sensor evidence via weighted noisy-OR."
      }
    ],
    "raw_payload": {
      "request": {
        "request_id": "ceeef0a4-2d57-4857-bf2d-113f5b2e9d70",
        "scenario": "output",
        "stage": "output",
        "prompt": "A harmless product photo of a ceramic mug on a desk.",
        "image_path": null,
        "image_base64": null,
        "metadata": {}
      },
      "response": {
        "request_id": "ceeef0a4-2d57-4857-bf2d-113f5b2e9d70",
        "scenario": "output",
        "stage": "output",
        "verdict": "allow",
        "categories": [],
        "confidence": 0.0,
        "reason": "Allowed because no policy signal reached the review threshold.",
        "evidence": {},
        "audit": {
          "audit_id": "0a0cabd1-3fd8-476d-a40b-7c5d71a51e95",
          "reason_code": "no_policy_signal_above_review_threshold",
          "human_reason": "Allowed because no policy signal reached the review threshold.",
          "policy_version": "logging-demo",
          "thresholds": {
            "block": 0.85,
            "review": 0.55
          },
          "matched_categories": [],
          "decision_path": [
            {
              "step": "collect_scores",
              "scores": {},
              "sources": {},
              "agreement": {}
            },
            {
              "step": "apply_block_rules",
              "threshold": 0.85,
              "hard_block_matches": [],
              "soft_block_matches": []
            },
            {
              "step": "apply_review_rules",
              "threshold": 0.55,
              "review_matches": []
            },
            {
              "step": "final_verdict",
              "verdict": "allow",
              "reason_code": "no_policy_signal_above_review_threshold"
            }
          ],
          "system_trace_id": "8b011928-fb92-48db-bb12-b489278da62c"
        },
        "signals": [
          {
            "name": "text_guard",
            "status": "ok",
            "categories": {},
            "text": [],
            "reason": "No flagged terms in text.",
            "raw": {
              "mode": "lexicon",
              "lexicon_matches": {},
              "text_length": 52
            }
          },
          {
            "name": "policy_fusion",
            "status": "ok",
            "categories": {},
            "text": [],
            "reason": "Fused calibrated sensor evidence via weighted noisy-OR.",
            "raw": {
              "mode": "weighted_noisy_or",
              "contributions": {},
              "agreement": {},
              "escalation": {
                "attempted": false,
                "needed": false
              }
            }
          }
        ],
        "notes": []
      },
      "signals": [
        {
          "name": "text_guard",
          "status": "ok",
          "categories": {},
          "text": [],
          "reason": "No flagged terms in text.",
          "raw": {
            "mode": "lexicon",
            "lexicon_matches": {},
            "text_length": 52
          }
        },
        {
          "name": "policy_fusion",
          "status": "ok",
          "categories": {},
          "text": [],
          "reason": "Fused calibrated sensor evidence via weighted noisy-OR.",
          "raw": {
            "mode": "weighted_noisy_or",
            "contributions": {},
            "agreement": {},
            "escalation": {
              "attempted": false,
              "needed": false
            }
          }
        }
      ],
      "notes": []
    },
    "created_at": "2026-06-06T06:21:44.397603Z"
  },
  "moderation_response": {
    "request_id": "ceeef0a4-2d57-4857-bf2d-113f5b2e9d70",
    "scenario": "output",
    "stage": "output",
    "verdict": "allow",
    "categories": [],
    "confidence": 0.0,
    "reason": "Allowed because no policy signal reached the review threshold.",
    "evidence": {},
    "audit": {
      "audit_id": "0a0cabd1-3fd8-476d-a40b-7c5d71a51e95",
      "reason_code": "no_policy_signal_above_review_threshold",
      "human_reason": "Allowed because no policy signal reached the review threshold.",
      "policy_version": "logging-demo",
      "thresholds": {
        "block": 0.85,
        "review": 0.55
      },
      "matched_categories": [],
      "decision_path": [
        {
          "step": "collect_scores",
          "scores": {},
          "sources": {},
          "agreement": {}
        },
        {
          "step": "apply_block_rules",
          "threshold": 0.85,
          "hard_block_matches": [],
          "soft_block_matches": []
        },
        {
          "step": "apply_review_rules",
          "threshold": 0.55,
          "review_matches": []
        },
        {
          "step": "final_verdict",
          "verdict": "allow",
          "reason_code": "no_policy_signal_above_review_threshold"
        }
      ],
      "system_trace_id": "8b011928-fb92-48db-bb12-b489278da62c"
    },
    "signals": [
      {
        "name": "text_guard",
        "status": "ok",
        "categories": {},
        "text": [],
        "reason": "No flagged terms in text.",
        "raw": {
          "mode": "lexicon",
          "lexicon_matches": {},
          "text_length": 52
        }
      },
      {
        "name": "policy_fusion",
        "status": "ok",
        "categories": {},
        "text": [],
        "reason": "Fused calibrated sensor evidence via weighted noisy-OR.",
        "raw": {
          "mode": "weighted_noisy_or",
          "contributions": {},
          "agreement": {},
          "escalation": {
            "attempted": false,
            "needed": false
          }
        }
      }
    ],
    "notes": []
  }
}
```

Final result: `PASS`
