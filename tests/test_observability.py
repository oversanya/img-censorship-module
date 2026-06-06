from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from censor_guard.config import Settings
from censor_guard.observability import (
    BusinessAuditEvent,
    JsonlLogSink,
    ModerationLogger,
    ObservabilityError,
    RawPayloadRecord,
    SystemLogEvent,
)
from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest


ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
ROOT = Path(__file__).resolve().parents[1]


class RecordingSink:
    def __init__(self) -> None:
        self.writes: list[
            tuple[list[SystemLogEvent], BusinessAuditEvent, RawPayloadRecord]
        ] = []

    def write_request(
        self,
        system_events: list[SystemLogEvent],
        audit_event: BusinessAuditEvent,
        raw_payload: RawPayloadRecord,
    ) -> None:
        self.writes.append((system_events, audit_event, raw_payload))


class FailingSink:
    def write_request(
        self,
        system_events: list[SystemLogEvent],
        audit_event: BusinessAuditEvent,
        raw_payload: RawPayloadRecord,
    ) -> None:
        raise ObservabilityError("jsonl write failed")


def lightweight_settings() -> Settings:
    return Settings(
        enable_ocr=False,
        enable_visual_classifier=False,
        enable_explicit_detector=False,
        enable_policy_judge=False,
        enable_injection_revealer=False,
        enable_image_sanitizer=False,
        enable_robust_guard=False,
        enable_llava_guard=False,
        text_model_id="",
        policy_version="test-policy",
    )


class ObservabilityPipelineTests(unittest.TestCase):
    def test_pipeline_writes_system_events_and_one_business_audit_event(self) -> None:
        sink = RecordingSink()
        logger = ModerationLogger(sink=sink, policy_version="test-policy")
        pipeline = GuardrailPipeline(settings=lightweight_settings(), logger=logger)

        response = pipeline.moderate(
            ModerationRequest(
                scenario="output",
                stage="output",
                prompt="ordinary prompt",
                image_base64=ONE_PIXEL_PNG_BASE64,
            )
        )

        self.assertEqual(len(sink.writes), 1)
        system_events, audit_event, raw_payload = sink.writes[0]
        event_types = [event.event_type for event in system_events]
        completed_sensors = {
            event.sensor_name
            for event in system_events
            if event.event_type == "sensor.completed"
        }

        self.assertIn("moderation.started", event_types)
        self.assertIn("fusion.completed", event_types)
        self.assertIn("decision.completed", event_types)
        self.assertIn("moderation.completed", event_types)
        self.assertTrue(
            {
                "text_guard",
                "ocr_adapter",
                "visual_classifier",
                "explicit_content_detector",
                "llava_guard",
            }.issubset(completed_sensors)
        )
        self.assertEqual(audit_event.audit_id, response.audit.audit_id)
        self.assertEqual(audit_event.trace_id, response.audit.system_trace_id)
        self.assertEqual(audit_event.reason_code, response.audit.reason_code)
        self.assertEqual(audit_event.thresholds, response.audit.thresholds)
        self.assertEqual(raw_payload.payload["request"]["request_id"], response.request_id)
        self.assertIn("response", audit_event.raw_payload)

    def test_pipeline_raises_when_required_logging_fails(self) -> None:
        logger = ModerationLogger(sink=FailingSink(), policy_version="test-policy")
        pipeline = GuardrailPipeline(settings=lightweight_settings(), logger=logger)

        with self.assertRaises(ObservabilityError):
            pipeline.moderate(ModerationRequest(scenario="output", stage="output"))

    def test_jsonl_sink_writes_three_append_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sink = JsonlLogSink(tmp)
            event = SystemLogEvent(
                trace_id="trace-id",
                request_id="request-id",
                scenario="output",
                stage="output",
                event_type="moderation.started",
            )

            sink.write_request([event], self._audit_event(), self._raw_payload())

            system_rows = self._read_jsonl(Path(tmp) / "system.jsonl")
            audit_rows = self._read_jsonl(Path(tmp) / "business_audit.jsonl")
            raw_rows = self._read_jsonl(Path(tmp) / "raw_payloads.jsonl")
            self.assertEqual(system_rows[0]["event_type"], "moderation.started")
            self.assertEqual(audit_rows[0]["audit_id"], "audit-id")
            self.assertEqual(raw_rows[0]["request_id"], "request-id")

    def test_jsonl_sink_serializes_parallel_writers(self) -> None:
        request_count = 75
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)

            def write_one(index: int) -> None:
                request_id = f"request-{index}"
                trace_id = f"trace-{index}"
                sink = JsonlLogSink(log_dir)
                sink.write_request(
                    [
                        SystemLogEvent(
                            trace_id=trace_id,
                            request_id=request_id,
                            scenario="output",
                            stage="output",
                            event_type="moderation.started",
                        )
                    ],
                    self._audit_event(
                        audit_id=f"audit-{index}",
                        trace_id=trace_id,
                        request_id=request_id,
                    ),
                    self._raw_payload(trace_id=trace_id, request_id=request_id),
                )

            with ThreadPoolExecutor(max_workers=12) as executor:
                list(executor.map(write_one, range(request_count)))

            system_rows = self._read_jsonl(log_dir / "system.jsonl")
            audit_rows = self._read_jsonl(log_dir / "business_audit.jsonl")
            raw_rows = self._read_jsonl(log_dir / "raw_payloads.jsonl")
            expected_request_ids = {f"request-{index}" for index in range(request_count)}

            self.assertEqual(len(system_rows), request_count)
            self.assertEqual(len(audit_rows), request_count)
            self.assertEqual(len(raw_rows), request_count)
            self.assertEqual({row["request_id"] for row in system_rows}, expected_request_ids)
            self.assertEqual({row["request_id"] for row in audit_rows}, expected_request_ids)
            self.assertEqual({row["request_id"] for row in raw_rows}, expected_request_ids)

    def test_jsonl_sink_serializes_parallel_processes(self) -> None:
        request_count = 16
        worker_code = """
import sys
from censor_guard.observability import BusinessAuditEvent, JsonlLogSink, RawPayloadRecord, SystemLogEvent

log_dir = sys.argv[1]
index = sys.argv[2]
request_id = f"process-request-{index}"
trace_id = f"process-trace-{index}"
JsonlLogSink(log_dir).write_request(
    [
        SystemLogEvent(
            trace_id=trace_id,
            request_id=request_id,
            scenario="output",
            stage="output",
            event_type="moderation.started",
        )
    ],
    BusinessAuditEvent(
        audit_id=f"process-audit-{index}",
        trace_id=trace_id,
        request_id=request_id,
        scenario="output",
        stage="output",
        verdict="allow",
        reason_code="no_policy_signal_above_review_threshold",
        human_reason="Allowed.",
    ),
    RawPayloadRecord(trace_id=trace_id, request_id=request_id),
)
"""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)

            def run_worker(index: int) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-c", worker_code, str(log_dir), str(index)],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(run_worker, range(request_count)))

            for result in results:
                self.assertEqual(result.returncode, 0, result.stderr)

            system_rows = self._read_jsonl(log_dir / "system.jsonl")
            audit_rows = self._read_jsonl(log_dir / "business_audit.jsonl")
            raw_rows = self._read_jsonl(log_dir / "raw_payloads.jsonl")
            expected_request_ids = {
                f"process-request-{index}" for index in range(request_count)
            }

            self.assertEqual(len(system_rows), request_count)
            self.assertEqual(len(audit_rows), request_count)
            self.assertEqual(len(raw_rows), request_count)
            self.assertEqual({row["request_id"] for row in system_rows}, expected_request_ids)
            self.assertEqual({row["request_id"] for row in audit_rows}, expected_request_ids)
            self.assertEqual({row["request_id"] for row in raw_rows}, expected_request_ids)

    def _audit_event(
        self,
        audit_id: str = "audit-id",
        trace_id: str = "trace-id",
        request_id: str = "request-id",
    ) -> BusinessAuditEvent:
        return BusinessAuditEvent(
            audit_id=audit_id,
            trace_id=trace_id,
            request_id=request_id,
            scenario="output",
            stage="output",
            verdict="allow",
            reason_code="no_policy_signal_above_review_threshold",
            human_reason="Allowed.",
        )

    def _raw_payload(
        self,
        trace_id: str = "trace-id",
        request_id: str = "request-id",
    ) -> RawPayloadRecord:
        return RawPayloadRecord(trace_id=trace_id, request_id=request_id)

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


if __name__ == "__main__":
    unittest.main()
