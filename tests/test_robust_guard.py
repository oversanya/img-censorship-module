from __future__ import annotations

import unittest

from censor_guard.schemas import SignalResult

try:
    from censor_guard.pipeline import GuardrailPipeline
except Exception as exc:  # pragma: no cover - optional deps (e.g. promptscreen) absent
    GuardrailPipeline = None
    _PIPELINE_IMPORT_ERROR = exc


class _FakeRobustGuard:
    """Подменяет настоящий RobustGuardAdapter, чтобы тестировать логику расхождения
    без robustbench/весов/тяжёлой модели."""

    def __init__(self, score: float | None, status: str = "ok", threshold: float = 0.02) -> None:
        self.score = score
        self.status = status
        self.threshold = threshold

    def moderate(self, image) -> SignalResult:
        if self.status != "ok":
            return SignalResult(name="robust_guard", status=self.status, reason="unavailable")
        return SignalResult(
            name="robust_guard",
            status="ok",
            raw={"robust_unsafe": self.score, "robust_threshold": self.threshold},
        )


@unittest.skipIf(GuardrailPipeline is None, "pipeline deps unavailable (e.g. promptscreen)")
class CheckAdversarialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = GuardrailPipeline()
        self.pipeline.robust_unsafe_min = 0.7
        self.pipeline.review_threshold = 0.55

    def test_blocks_when_robust_unsafe_but_main_passes(self) -> None:
        # Направленный обход: робастная уверенно unsafe (0.95), основная пропускает (0.1).
        self.pipeline.robust_guard = _FakeRobustGuard(score=0.95)
        result = self.pipeline.check_adversarial(image=object(), main_unsafe_score=0.1)
        self.assertEqual(result["verdict"], "block")

    def test_passes_when_main_already_flags(self) -> None:
        # Робастная unsafe, но и основная уже видит unsafe (0.8 ≥ review) — не adversarial.
        self.pipeline.robust_guard = _FakeRobustGuard(score=0.95)
        result = self.pipeline.check_adversarial(image=object(), main_unsafe_score=0.8)
        self.assertEqual(result["verdict"], "pass")

    def test_passes_when_robust_thinks_safe(self) -> None:
        # Робастная тоже считает safe (0.2 < min) — расхождения нет.
        self.pipeline.robust_guard = _FakeRobustGuard(score=0.2)
        result = self.pipeline.check_adversarial(image=object(), main_unsafe_score=0.1)
        self.assertEqual(result["verdict"], "pass")

    def test_does_not_block_reverse_divergence(self) -> None:
        # Обратное расхождение (основная паникует, робастная спокойна) не штрафуем.
        self.pipeline.robust_guard = _FakeRobustGuard(score=0.05)
        result = self.pipeline.check_adversarial(image=object(), main_unsafe_score=0.9)
        self.assertEqual(result["verdict"], "pass")

    def test_passes_when_robust_unavailable(self) -> None:
        # Нет robustbench/весов → индикатор молчит, блок не навязываем.
        self.pipeline.robust_guard = _FakeRobustGuard(score=None, status="skipped")
        result = self.pipeline.check_adversarial(image=object(), main_unsafe_score=0.1)
        self.assertEqual(result["verdict"], "pass")


class RobustGuardAdapterTests(unittest.TestCase):
    def test_skipped_when_disabled(self) -> None:
        from censor_guard.adapters.robust_guard import RobustGuardAdapter

        adapter = RobustGuardAdapter(enabled=False, probe_path="robust_probe.npz", model_dir="models")
        self.assertEqual(adapter.moderate(object()).status, "skipped")
        self.assertIsNone(adapter.unsafe_score(object()))

    def test_soft_degrades_on_missing_weights(self) -> None:
        from censor_guard.adapters.robust_guard import RobustGuardAdapter

        adapter = RobustGuardAdapter(enabled=True, probe_path="does_not_exist.npz", model_dir="models")
        signal = adapter.moderate(object())
        self.assertEqual(signal.status, "skipped")  # missing weights = soft unavailable
        self.assertIsNone(adapter.unsafe_score(object()))


if __name__ == "__main__":
    unittest.main()
