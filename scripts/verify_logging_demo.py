from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LOG_DIR = ROOT / "logs" / "logging_demo"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_pipeline(log_dir: Path):
    from censor_guard.config import Settings
    from censor_guard.pipeline import GuardrailPipeline

    settings = Settings(
        log_dir=str(log_dir),
        policy_version="logging-demo",
        enable_ocr=False,
        enable_visual_classifier=False,
        enable_explicit_detector=False,
        enable_policy_judge=False,
        enable_injection_revealer=False,
        enable_image_sanitizer=False,
        enable_robust_guard=False,
        enable_llava_guard=False,
        text_model_id="",
    )
    return GuardrailPipeline(settings=settings)


def run_verification(log_dir: Path, clean: bool = False) -> dict[str, Any]:
    from censor_guard.schemas import ModerationRequest

    if clean and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pipeline = make_pipeline(log_dir)
    request = ModerationRequest(
        scenario="output",
        stage="output",
        prompt="A harmless product photo of a ceramic mug on a desk.",
    )
    response = pipeline.moderate(request)
    request_id = response.request_id

    system_rows = [
        row for row in read_jsonl(log_dir / "system.jsonl")
        if row.get("request_id") == request_id
    ]
    audit_rows = [
        row for row in read_jsonl(log_dir / "business_audit.jsonl")
        if row.get("request_id") == request_id
    ]
    raw_rows = [
        row for row in read_jsonl(log_dir / "raw_payloads.jsonl")
        if row.get("request_id") == request_id
    ]

    checks = {
        "response_has_audit": bool(response.audit.reason_code),
        "system_jsonl_exists": (log_dir / "system.jsonl").is_file(),
        "business_audit_jsonl_exists": (log_dir / "business_audit.jsonl").is_file(),
        "raw_payloads_jsonl_exists": (log_dir / "raw_payloads.jsonl").is_file(),
        "system_jsonl_rows_gte_1": len(system_rows) >= 1,
        "business_audit_jsonl_rows_eq_1": len(audit_rows) == 1,
        "raw_payloads_jsonl_rows_eq_1": len(raw_rows) == 1,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "log_dir": str(log_dir),
        "request_id": request_id,
        "verdict": response.verdict,
        "audit_reason_code": response.audit.reason_code,
        "audit_system_trace_id": response.audit.system_trace_id,
        "counts": {
            "system_jsonl_rows": len(system_rows),
            "business_audit_jsonl_rows": len(audit_rows),
            "raw_payloads_jsonl_rows": len(raw_rows),
        },
        "checks": checks,
        "audit_sample": audit_rows[0] if audit_rows else None,
        "moderation_response": response.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify moderation logging against JSONL files.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    try:
        summary = run_verification(Path(args.log_dir), clean=args.clean)
    except Exception as exc:
        summary = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
