from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.llava_guard import LlavaGuardAdapter
from censor_guard.adapters.ocr import OCRAdapter
from censor_guard.adapters.policy_judge import PolicyJudge
from censor_guard.adapters.robust_guard import RobustGuardAdapter
from censor_guard.adapters.text_classifier import TextGuard
from censor_guard.adapters.visual_classifier import VisualClassifierAdapter
from censor_guard.config import Settings
from censor_guard.decision import DecisionEngine
from censor_guard.image_utils import load_image
from censor_guard.observability import DecisionExplanation, ModerationLogger, ModerationTrace
from censor_guard.schemas import ModerationRequest, ModerationResponse, SignalResult


T = TypeVar("T")


class GuardrailPipeline:
    """Orchestrates moderation sensors, robust guardrails, decisions, and mandatory logging."""

    def __init__(self, settings: Settings | None = None, logger: ModerationLogger | None = None) -> None:
        self.settings = settings or Settings()
        self.logger = logger or ModerationLogger.from_settings(self.settings)
        self.text_guard = TextGuard(
            enabled=self.settings.enable_text_guard,
            model_id=self.settings.text_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.ocr = OCRAdapter(
            enabled=self.settings.enable_ocr,
            tesseract_cmd=self.settings.tesseract_cmd,
            tessdata_dir=self.settings.tessdata_dir,
        )
        self.visual = VisualClassifierAdapter(
            enabled=self.settings.enable_visual_classifier,
            model_id=self.settings.visual_model_id,
            cache_dir=self.settings.hf_cache_dir,
            calibration_floor=self.settings.calibration_floor,
        )
        self.explicit = ExplicitContentAdapter(
            enabled=self.settings.enable_explicit_detector,
            model_id=self.settings.explicit_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.llava_guard = LlavaGuardAdapter(
            enabled=self.settings.enable_llava_guard,
            model_id=self.settings.llava_guard_model_id,
            cache_dir=self.settings.hf_cache_dir,
            max_new_tokens=self.settings.llava_guard_max_new_tokens,
            unsafe_score=self.settings.llava_guard_unsafe_score,
        )
        self.policy_judge = PolicyJudge(
            enabled=self.settings.enable_policy_judge,
            model_id=self.settings.policy_judge_model_id,
            review_threshold=self.settings.review_threshold,
            block_threshold=self.settings.block_threshold,
        )
        self.decision_engine = DecisionEngine(
            block_threshold=self.settings.block_threshold,
            review_threshold=self.settings.review_threshold,
            policy_version=self.settings.policy_version,
        )
        self.string_guard, self._string_guard_error = self._build_string_guard()
        self.image_analyzer, self._image_analyzer_error = self._build_image_analyzer()
        self.robust_guard = RobustGuardAdapter(
            enabled=self.settings.enable_robust_guard,
            probe_path=self.settings.robust_probe_path,
            model_dir=self.settings.robust_model_dir,
        )
        self.robust_unsafe_min = self.settings.robust_unsafe_min
        self.review_threshold = self.settings.review_threshold
        self.use_guardrails = True

    def assess(
        self,
        image,
        prompt: str | None,
        stage: str,
        trace: ModerationTrace | None = None,
        request: ModerationRequest | None = None,
    ) -> list[SignalResult]:
        signals = [
            self._capture(
                trace=trace,
                request=request,
                sensor_name=self.text_guard.name,
                fn=lambda: self.text_guard.moderate(prompt),
            )
        ]

        if image is not None:
            ocr_signal = self._capture(
                trace=trace,
                request=request,
                sensor_name=self.ocr.name,
                fn=lambda: self.ocr.extract(image),
            )
            signals.append(ocr_signal)

            if ocr_signal.text:
                ocr_text = "\n".join(ocr_signal.text)
                signals.append(
                    self._capture(
                        trace=trace,
                        request=request,
                        sensor_name="ocr_text_guard",
                        fn=lambda: self._moderate_ocr_text(ocr_text),
                    )
                )

            signals.append(
                self._capture(
                    trace=trace,
                    request=request,
                    sensor_name=self.visual.name,
                    fn=lambda: self.visual.moderate(image),
                )
            )
            signals.append(
                self._capture(
                    trace=trace,
                    request=request,
                    sensor_name=self.explicit.name,
                    fn=lambda: self.explicit.moderate(image),
                )
            )
            signals.append(
                self._capture(
                    trace=trace,
                    request=request,
                    sensor_name=self.llava_guard.name,
                    fn=lambda: self.llava_guard.moderate(image),
                )
            )

        fusion_started = time.perf_counter()
        policy_signals = self.policy_judge.judge(
            image=image,
            prompt=prompt,
            signals=signals,
            stage=stage,
        )
        if trace is not None and request is not None:
            self.logger.system_event(
                trace=trace,
                request=request,
                event_type="fusion.completed",
                status="ok",
                duration_ms=round((time.perf_counter() - fusion_started) * 1000, 3),
                payload={"signals": [signal.model_dump(mode="json") for signal in policy_signals]},
            )
        signals.extend(policy_signals)
        return signals

    def get_decision(
        self,
        request: ModerationRequest,
        image=None,
        prompt: str | None = None,
        trace: ModerationTrace | None = None,
    ) -> ModerationResponse:
        if image is None:
            image = load_image(request)
        if prompt is None:
            prompt = request.prompt
        signals = self.assess(image, prompt, request.stage, trace=trace, request=request)
        return self.decision_engine.decide(request, signals)

    def compare_guarded(
        self,
        request: ModerationRequest,
        image=None,
        trace: ModerationTrace | None = None,
    ) -> dict[str, Any]:
        if image is None:
            image = load_image(request)
        resp_before = self.get_decision(request, image=image, trace=trace)
        conf_before = resp_before.confidence

        if image is None or self.image_analyzer is None:
            return {
                "verdict": "pass",
                "reason": self._image_guardrail_skip_reason(image),
                "conf_before": conf_before,
                "conf_after": conf_before,
                "decision_after": resp_before,
            }

        sanitized = self._capture(
            trace=trace,
            request=request,
            sensor_name="image_sanitizer",
            fn=lambda: self.image_analyzer.process(image)["image"],
        )
        resp_after = self.get_decision(request, image=sanitized, trace=trace)
        conf_after = resp_after.confidence

        if resp_after.verdict != resp_before.verdict and abs(conf_before - conf_after) > 0.4:
            return {
                "verdict": "block",
                "reason": "suspicious image detected, highly unstable",
                "conf_before": conf_before,
                "conf_after": conf_after,
                "decision_after": resp_after,
            }

        return {
            "verdict": "pass",
            "reason": "no inconsistencies revealed",
            "conf_before": conf_before,
            "conf_after": conf_after,
            "decision_after": resp_after,
        }

    def check_adversarial(
        self,
        image,
        main_unsafe_score: float,
        trace: ModerationTrace | None = None,
        request: ModerationRequest | None = None,
    ) -> dict[str, Any]:
        robust_signal = self._capture(
            trace=trace,
            request=request,
            sensor_name=getattr(self.robust_guard, "name", "robust_guard"),
            fn=lambda: self.robust_guard.moderate(image),
        )
        if robust_signal.status != "ok":
            return {"verdict": "pass", "reason": "robust guard unavailable", "signal": robust_signal}

        robust_unsafe = float(robust_signal.raw["robust_unsafe"])
        main_unsafe = float(main_unsafe_score)
        divergence = robust_unsafe - main_unsafe
        is_adversarial = robust_unsafe >= self.robust_unsafe_min and main_unsafe < self.review_threshold

        result: dict[str, Any] = {
            "robust_unsafe": robust_unsafe,
            "main_unsafe": main_unsafe,
            "divergence": divergence,
            "signal": robust_signal,
        }
        if is_adversarial:
            result["verdict"] = "block"
            result["reason"] = (
                "adversarial example suspected: robust model flags unsafe while main "
                f"pipeline passes (robust P(unsafe)={robust_unsafe:.3f} >= {self.robust_unsafe_min}, "
                f"main unsafe={main_unsafe:.3f} < {self.review_threshold})"
            )
        else:
            result["verdict"] = "pass"
            result["reason"] = "robust model does not contradict main pipeline"
        return result

    def check_text_injections(self, prompt: str | None) -> dict[str, Any]:
        if self.string_guard is None:
            return {
                "status": "skipped",
                "safe": True,
                "reasons": [self._string_guard_error or "Prompt injection scanner disabled by configuration."],
                "confidence": {},
                "raw": {},
            }
        return self.string_guard.process(prompt or "")

    def moderate(self, request: ModerationRequest) -> ModerationResponse:
        trace = self.logger.start_trace(request)
        use_guardrails = self.use_guardrails and request.stage != "output"

        if request.stage == "input":
            prompt_analysis = self._capture(
                trace=trace,
                request=request,
                sensor_name="string_guard",
                fn=lambda: self.check_text_injections(request.prompt),
            )
            if prompt_analysis.get("safe") is False:
                response = self._guardrail_response(
                    request=request,
                    reason_code="input_prompt_guardrail_block",
                    reason="; ".join(prompt_analysis.get("reasons", [])) or "Input prompt guardrail blocked the request.",
                    confidence=self._max_confidence(prompt_analysis.get("confidence", {})),
                    notes=["input prompt guardrail"],
                    signals=[
                        SignalResult(
                            name="string_guard",
                            status="ok",
                            reason="Input prompt guardrail blocked the request.",
                            raw=self._safe_guardrail_raw(prompt_analysis),
                        )
                    ],
                )
                return self._finalize(trace, request, response)

        image = load_image(request)
        if not use_guardrails:
            response = self.get_decision(request, image=image, trace=trace)
            return self._finalize(trace, request, response)

        inconsistency_test = self.compare_guarded(request, image=image, trace=trace)

        if image is not None:
            adversarial_test = self.check_adversarial(
                image,
                inconsistency_test["conf_before"],
                trace=trace,
                request=request,
            )
            if adversarial_test["verdict"] == "block":
                response = self._guardrail_response(
                    request=request,
                    reason_code="robust_adversarial_guardrail_block",
                    reason=adversarial_test["reason"],
                    confidence=round(float(adversarial_test["divergence"]), 4),
                    notes=["robust-model adversarial guardrail"],
                    signals=[adversarial_test["signal"]],
                )
                return self._finalize(trace, request, response)

        if inconsistency_test["verdict"] == "block":
            response = self._guardrail_response(
                request=request,
                reason_code="image_sanitizer_instability_block",
                reason=inconsistency_test["reason"],
                confidence=round(abs(inconsistency_test["conf_before"] - inconsistency_test["conf_after"]), 4),
                notes=["input image guardrailing"],
                signals=list(inconsistency_test["decision_after"].signals),
            )
            return self._finalize(trace, request, response)

        return self._finalize(trace, request, inconsistency_test["decision_after"])

    def _capture(
        self,
        trace: ModerationTrace | None,
        request: ModerationRequest | None,
        sensor_name: str,
        fn: Callable[[], T],
    ) -> T:
        if trace is None or request is None:
            return fn()
        return self.logger.capture_sensor(
            trace=trace,
            request=request,
            sensor_name=sensor_name,
            fn=fn,
        )

    def _moderate_ocr_text(self, text: str) -> SignalResult:
        result = self.text_guard.moderate(text)
        result.name = "ocr_text_guard"
        return result

    def _guardrail_response(
        self,
        request: ModerationRequest,
        reason_code: str,
        reason: str,
        confidence: float,
        notes: list[str],
        signals: list[SignalResult],
    ) -> ModerationResponse:
        audit = DecisionExplanation(
            reason_code=reason_code,
            human_reason=reason,
            policy_version=self.settings.policy_version,
            thresholds={
                "block": self.settings.block_threshold,
                "review": self.settings.review_threshold,
            },
            decision_path=[
                {
                    "step": "guardrail_block",
                    "reason_code": reason_code,
                    "notes": notes,
                    "confidence": round(confidence, 4),
                }
            ],
        )
        return ModerationResponse(
            request_id=request.request_id,
            scenario=request.scenario,
            stage=request.stage,
            verdict="block",
            categories=[],
            confidence=round(confidence, 4),
            reason=reason,
            evidence={},
            audit=audit,
            signals=signals,
            notes=notes,
        )

    def _finalize(
        self,
        trace: ModerationTrace,
        request: ModerationRequest,
        response: ModerationResponse,
    ) -> ModerationResponse:
        self.logger.system_event(
            trace=trace,
            request=request,
            event_type="decision.completed",
            status=response.verdict,
            payload={"audit": response.audit.model_dump(mode="json")},
        )
        self.logger.commit_request(trace, request, response)
        return response

    def _max_confidence(self, values: dict[str, float] | None) -> float:
        if not values:
            return 1.0
        return max(float(value) for value in values.values())

    def _safe_guardrail_raw(self, value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: val
            for key, val in value.items()
            if key != "raw"
        }

    def _build_string_guard(self):
        if not self.settings.enable_injection_revealer:
            return None, "Prompt injection scanner disabled by configuration."
        try:
            from censor_guard.guardrails.string_guard import StringGuard
        except Exception as exc:
            return None, f"Prompt injection scanner unavailable: {exc}"
        return StringGuard(enabled=True), None

    def _build_image_analyzer(self):
        if not self.settings.enable_image_sanitizer:
            return None, "Image sanitizer disabled by configuration."
        try:
            from censor_guard.guardrails.image_guard import ImageAnalyzer
        except Exception as exc:
            return None, f"Image sanitizer unavailable: {exc}"
        return ImageAnalyzer(enabled=True), None

    def _image_guardrail_skip_reason(self, image) -> str:
        if image is None:
            return "image guardrail skipped: no image"
        return self._image_analyzer_error or "image guardrail skipped"
