from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field


T = TypeVar("T")
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


class ObservabilityError(RuntimeError):
    """Raised when required moderation logging cannot be completed."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DecisionExplanation(BaseModel):
    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    reason_code: str
    human_reason: str
    policy_version: str
    thresholds: dict[str, float] = Field(default_factory=dict)
    matched_categories: list[dict[str, Any]] = Field(default_factory=list)
    decision_path: list[dict[str, Any]] = Field(default_factory=list)
    system_trace_id: str | None = None


class SystemLogEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    request_id: str
    scenario: str
    stage: str
    event_type: str
    timestamp: datetime = Field(default_factory=_utcnow)
    sensor_name: str | None = None
    status: str | None = None
    duration_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    exception: dict[str, Any] | None = None


class BusinessAuditEvent(BaseModel):
    audit_id: str
    trace_id: str
    request_id: str
    scenario: str
    stage: str
    verdict: str
    reason_code: str
    human_reason: str
    categories: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    thresholds: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    fusion_contributions: dict[str, Any] = Field(default_factory=dict)
    signals_summary: list[dict[str, Any]] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class RawPayloadRecord(BaseModel):
    payload_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    request_id: str
    payload_type: str = "moderation_full_raw"
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ModerationTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    system_events: list[SystemLogEvent] = Field(default_factory=list)


class LogSink(Protocol):
    def write_request(
        self,
        system_events: list[SystemLogEvent],
        audit_event: BusinessAuditEvent,
        raw_payload: RawPayloadRecord,
    ) -> None:
        ...


class _JsonlWriteLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock: threading.Lock | None = None
        self._file = None
        self._locked = False

    def __enter__(self) -> "_JsonlWriteLock":
        key = str(self.path.resolve())
        with _LOCKS_GUARD:
            thread_lock = _LOCKS.setdefault(key, threading.Lock())
        thread_lock.acquire()
        self._thread_lock = thread_lock

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a+b")
            self._file.seek(0, 2)
            if self._file.tell() == 0:
                self._file.write(b"\0")
                self._file.flush()
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            self._locked = True
            return self
        except Exception:
            if self._file is not None:
                self._file.close()
            thread_lock.release()
            self._thread_lock = None
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._file is not None and self._locked:
                self._file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            if self._file is not None:
                self._file.close()
            if self._thread_lock is not None:
                self._thread_lock.release()


class JsonlLogSink:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.system_path = self.log_dir / "system.jsonl"
        self.audit_path = self.log_dir / "business_audit.jsonl"
        self.raw_payload_path = self.log_dir / "raw_payloads.jsonl"
        self.lock_path = self.log_dir / ".jsonl.lock"

    def write_request(
        self,
        system_events: list[SystemLogEvent],
        audit_event: BusinessAuditEvent,
        raw_payload: RawPayloadRecord,
    ) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with _JsonlWriteLock(self.lock_path):
                self._append_lines(self.system_path, system_events)
                self._append_lines(self.raw_payload_path, [raw_payload])
                self._append_lines(self.audit_path, [audit_event])
        except OSError as exc:
            raise ObservabilityError(f"JSONL logging failed: {exc}") from exc

    def _append_lines(self, path: Path, rows: list[BaseModel]) -> None:
        if not rows:
            return
        with path.open("a", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False))
                file.write("\n")


class ModerationLogger:
    def __init__(self, sink: LogSink, policy_version: str) -> None:
        self.sink = sink
        self.policy_version = policy_version

    @classmethod
    def from_settings(cls, settings: Any) -> "ModerationLogger":
        return cls(
            sink=JsonlLogSink(settings.log_dir),
            policy_version=settings.policy_version,
        )

    def start_trace(self, request: Any) -> ModerationTrace:
        trace = ModerationTrace()
        self.system_event(
            trace=trace,
            request=request,
            event_type="moderation.started",
            payload={"request": self._dump(request)},
        )
        return trace

    def system_event(
        self,
        trace: ModerationTrace,
        request: Any,
        event_type: str,
        sensor_name: str | None = None,
        status: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
        exception: dict[str, Any] | None = None,
    ) -> None:
        trace.system_events.append(
            SystemLogEvent(
                trace_id=trace.trace_id,
                request_id=request.request_id,
                scenario=request.scenario,
                stage=request.stage,
                event_type=event_type,
                sensor_name=sensor_name,
                status=status,
                duration_ms=duration_ms,
                payload=payload or {},
                exception=exception,
            )
        )

    def capture_sensor(
        self,
        trace: ModerationTrace,
        request: Any,
        sensor_name: str,
        fn: Callable[[], T],
    ) -> T:
        self.system_event(
            trace=trace,
            request=request,
            event_type="sensor.started",
            sensor_name=sensor_name,
        )
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self.system_event(
                trace=trace,
                request=request,
                event_type="sensor.failed",
                sensor_name=sensor_name,
                status="error",
                duration_ms=duration_ms,
                exception={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        self.system_event(
            trace=trace,
            request=request,
            event_type="sensor.completed",
            sensor_name=sensor_name,
            status=getattr(result, "status", None),
            duration_ms=duration_ms,
            payload={"result": self._dump(result)},
        )
        return result

    def commit_request(self, trace: ModerationTrace, request: Any, response: Any) -> None:
        response.audit.system_trace_id = trace.trace_id
        self.system_event(
            trace=trace,
            request=request,
            event_type="moderation.completed",
            status=response.verdict,
            payload={"response": self._dump(response)},
        )
        raw_payload = RawPayloadRecord(
            trace_id=trace.trace_id,
            request_id=request.request_id,
            payload=self._raw_payload(request, response),
        )
        audit_event = self._business_audit_event(trace, request, response, raw_payload)
        self.sink.write_request(trace.system_events, audit_event, raw_payload)

    def _business_audit_event(
        self,
        trace: ModerationTrace,
        request: Any,
        response: Any,
        raw_payload: RawPayloadRecord,
    ) -> BusinessAuditEvent:
        return BusinessAuditEvent(
            audit_id=response.audit.audit_id,
            trace_id=trace.trace_id,
            request_id=request.request_id,
            scenario=request.scenario,
            stage=request.stage,
            verdict=response.verdict,
            reason_code=response.audit.reason_code,
            human_reason=response.audit.human_reason,
            categories=response.categories,
            confidence=response.confidence,
            thresholds=response.audit.thresholds,
            evidence=response.evidence,
            fusion_contributions=self._fusion_contributions(response),
            signals_summary=self._signals_summary(response),
            raw_payload=raw_payload.payload,
        )

    def _raw_payload(self, request: Any, response: Any) -> dict[str, Any]:
        return {
            "request": self._dump(request),
            "response": self._dump(response),
            "signals": [self._dump(signal) for signal in response.signals],
            "notes": list(response.notes),
        }

    def _fusion_contributions(self, response: Any) -> dict[str, Any]:
        for signal in response.signals:
            if signal.name == "policy_fusion":
                return signal.raw.get("contributions", {})
        return {}

    def _signals_summary(self, response: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": signal.name,
                "status": signal.status,
                "categories": dict(signal.categories),
                "text": list(signal.text),
                "reason": signal.reason,
            }
            for signal in response.signals
        ]

    def _dump(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value
