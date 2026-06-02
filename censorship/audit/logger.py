"""Audit logger — append-only jsonlines writer for all pipeline verdicts."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Union

from censorship.core.verdict import Verdict

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Thread-safe audit logger.
    Writes one JSON object per line to a .jsonl file.
    Format is SIEM/ELK-compatible.
    """

    def __init__(self, log_path: Union[str, Path] = "audit.jsonl"):
        self.log_path = Path(log_path)
        self._lock = threading.Lock()

    def log(self, verdict: Verdict, user_id: str | None = None) -> None:
        record = {
            "image_id": verdict.image_id,
            "timestamp": verdict.timestamp,
            "decision": verdict.decision,
            "primary_category": verdict.primary_category,
            "confidence": verdict.reasoner_confidence,
            "classifier": verdict.classifier_model,
            "reasoner": verdict.reasoner_model,
            "classifier_scores": verdict.classifier_scores,
            "classifier_triggered": verdict.classifier_triggered,
            "latency_ms": round(verdict.latency_ms, 2),
            "user_id": user_id,
            "pipeline_version": verdict.pipeline_version,
            "prompt_verdict": verdict.prompt_verdict,
            "prompt_category": verdict.prompt_category,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.error(f"Failed to write audit log: {e}")

    def read_all(self) -> list[dict]:
        """Read all audit records (for analysis/testing)."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
