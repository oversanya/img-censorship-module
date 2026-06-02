"""End-to-end pipeline tests using mock classifiers and reasoners.

No GPU or real models required — all ML calls are mocked.
"""

import io
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from censorship.core.verdict import ClassifierResult, ReasonerResult, PromptGuardResult, Verdict
from censorship.core.taxonomy import TaxonomyLoader
from censorship.core.policy import PolicyEngine
from censorship.classifiers.base import ImageClassifier
from censorship.reasoners.base import ImageReasoner
from censorship.prompt_guard.base import PromptGuard
from censorship.pipeline.image_pipeline import ImagePipeline
from censorship.pipeline.combined_pipeline import CombinedPipeline
from censorship.audit.logger import AuditLogger


TAXONOMY_PATH = Path(__file__).parent.parent / "config" / "taxonomy.yaml"
POLICY_PATH = Path(__file__).parent.parent / "config" / "policy_bank.yaml"


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class MockClassifier(ImageClassifier):
    model_name = "mock-classifier"
    supported_categories = ["sexual_explicit", "violence_gore", "extremism"]

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, image_path) -> ClassifierResult:
        self._loaded = True
        triggered = [c for c, s in self._scores.items() if s >= 0.50]
        return ClassifierResult(
            model=self.model_name,
            scores=self._scores.copy(),
            is_unsafe=any(s >= 0.50 for s in self._scores.values()),
            triggered_categories=triggered,
            latency_ms=5.0,
        )


class MockReasoner(ImageReasoner):
    model_name = "mock-reasoner"

    def __init__(self, verdict: str, category: str | None, confidence: float, rationale: str = "Mock rationale."):
        self._verdict = verdict
        self._category = category
        self._confidence = confidence
        self._rationale = rationale
        self._loaded = False
        self.call_count = 0

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def reason(self, image_path, policy_text: str) -> ReasonerResult:
        self.call_count += 1
        return ReasonerResult(
            model=self.model_name,
            verdict=self._verdict,
            category=self._category,
            confidence=self._confidence,
            rationale=self._rationale,
            latency_ms=10.0,
        )


class MockPromptGuard(PromptGuard):
    model_name = "mock-prompt-guard"

    def __init__(self, verdict: str, category: str | None = None):
        self._verdict = verdict
        self._category = category

    def load(self) -> None: pass

    def check(self, text: str) -> PromptGuardResult:
        return PromptGuardResult(
            model=self.model_name,
            verdict=self._verdict,
            category=self._category,
            confidence=0.90,
            rationale="Mock rationale.",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def taxonomy():
    return TaxonomyLoader(TAXONOMY_PATH)


@pytest.fixture
def policy(taxonomy):
    return PolicyEngine(taxonomy, POLICY_PATH)


@pytest.fixture
def safe_image_path(tmp_path):
    """Create a small JPEG fixture for tests."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    p = tmp_path / "safe.jpg"
    img.save(p, format="JPEG")
    return p


@pytest.fixture
def audit_log_path(tmp_path):
    return tmp_path / "test_audit.jsonl"


def make_pipeline(
    classifier_scores,
    reasoner_result=None,
    taxonomy=None,
    policy=None,
    audit_log_path=None,
):
    if taxonomy is None:
        taxonomy = TaxonomyLoader(TAXONOMY_PATH)
    if policy is None:
        policy = PolicyEngine(taxonomy, POLICY_PATH)

    clf = MockClassifier(scores=classifier_scores)
    rsn = None
    if reasoner_result:
        rsn = MockReasoner(*reasoner_result)

    return ImagePipeline(
        classifier=clf,
        reasoner=rsn,
        taxonomy=taxonomy,
        policy_engine=policy,
        audit_logger=AuditLogger(audit_log_path) if audit_log_path else None,
    )


# ---------------------------------------------------------------------------
# Tests: ALLOW path (low confidence)
# ---------------------------------------------------------------------------

class TestAllowPath:
    def test_low_confidence_is_allowed(self, safe_image_path):
        # Scores well below fast_threshold_allow (0.05) → immediate ALLOW
        pipeline = make_pipeline({"sexual_explicit": 0.02, "violence_gore": 0.01})
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "ALLOW"
        assert verdict.reasoner_model is None

    def test_allow_does_not_call_reasoner(self, safe_image_path):
        rsn = MockReasoner("BLOCK", "sexual_explicit", 0.95)
        clf = MockClassifier(scores={"sexual_explicit": 0.03})
        pipeline = ImagePipeline(
            classifier=clf,
            reasoner=rsn,
            taxonomy=TaxonomyLoader(TAXONOMY_PATH),
            policy_engine=PolicyEngine(TaxonomyLoader(TAXONOMY_PATH), POLICY_PATH),
        )
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "ALLOW"
        assert rsn.call_count == 0


# ---------------------------------------------------------------------------
# Tests: BLOCK path (high confidence, no reasoner)
# ---------------------------------------------------------------------------

class TestBlockPath:
    def test_high_confidence_blocks_immediately(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.95, "violence_gore": 0.10})
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "BLOCK"

    def test_block_sets_primary_category(self, safe_image_path):
        pipeline = make_pipeline({"violence_gore": 0.92, "sexual_explicit": 0.20})
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "BLOCK"
        assert verdict.primary_category == "violence_gore"

    def test_high_confidence_does_not_call_reasoner(self, safe_image_path):
        rsn = MockReasoner("ALLOW", None, 0.10)
        clf = MockClassifier(scores={"sexual_explicit": 0.95})
        pipeline = ImagePipeline(
            classifier=clf,
            reasoner=rsn,
            taxonomy=TaxonomyLoader(TAXONOMY_PATH),
            policy_engine=PolicyEngine(TaxonomyLoader(TAXONOMY_PATH), POLICY_PATH),
        )
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "BLOCK"
        assert rsn.call_count == 0

    def test_verdict_has_classifier_scores(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.94})
        verdict = pipeline.run(safe_image_path)
        assert "sexual_explicit" in verdict.classifier_scores
        assert verdict.classifier_scores["sexual_explicit"] == pytest.approx(0.94)


# ---------------------------------------------------------------------------
# Tests: Gray zone → reasoner called
# ---------------------------------------------------------------------------

class TestGrayZonePath:
    def test_gray_zone_calls_reasoner(self, safe_image_path):
        rsn = MockReasoner("BLOCK", "sexual_explicit", 0.82, "Explicit content detected.")
        clf = MockClassifier(scores={"sexual_explicit": 0.70})
        pipeline = ImagePipeline(
            classifier=clf,
            reasoner=rsn,
            taxonomy=TaxonomyLoader(TAXONOMY_PATH),
            policy_engine=PolicyEngine(TaxonomyLoader(TAXONOMY_PATH), POLICY_PATH),
        )
        verdict = pipeline.run(safe_image_path)
        assert rsn.call_count == 1
        assert verdict.reasoner_model == "mock-reasoner"
        assert verdict.reasoner_rationale == "Explicit content detected."

    def test_gray_zone_reasoner_allow(self, safe_image_path):
        rsn = MockReasoner("ALLOW", None, 0.15, "Content is acceptable.")
        clf = MockClassifier(scores={"sexual_explicit": 0.65})
        pipeline = ImagePipeline(
            classifier=clf,
            reasoner=rsn,
            taxonomy=TaxonomyLoader(TAXONOMY_PATH),
            policy_engine=PolicyEngine(TaxonomyLoader(TAXONOMY_PATH), POLICY_PATH),
        )
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "ALLOW"
        assert rsn.call_count == 1

    def test_gray_zone_no_reasoner_returns_review(self, safe_image_path):
        clf = MockClassifier(scores={"sexual_explicit": 0.70})
        pipeline = ImagePipeline(
            classifier=clf,
            reasoner=None,
            taxonomy=TaxonomyLoader(TAXONOMY_PATH),
            policy_engine=PolicyEngine(TaxonomyLoader(TAXONOMY_PATH), POLICY_PATH),
        )
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "REVIEW"


# ---------------------------------------------------------------------------
# Tests: Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_log_written(self, safe_image_path, audit_log_path):
        pipeline = make_pipeline(
            {"sexual_explicit": 0.95},
            audit_log_path=audit_log_path,
        )
        pipeline.run(safe_image_path, user_id="user_001")
        assert audit_log_path.exists()
        records = AuditLogger(audit_log_path).read_all()
        assert len(records) == 1
        assert records[0]["decision"] == "BLOCK"
        assert records[0]["user_id"] == "user_001"

    def test_audit_log_multiple_records(self, safe_image_path, audit_log_path):
        pipeline = make_pipeline(
            {"sexual_explicit": 0.02},
            audit_log_path=audit_log_path,
        )
        for i in range(3):
            pipeline.run(safe_image_path, user_id=f"user_{i}")
        records = AuditLogger(audit_log_path).read_all()
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Tests: Combined pipeline (prompt + image)
# ---------------------------------------------------------------------------

class TestCombinedPipeline:
    def test_prompt_block_short_circuits(self, safe_image_path):
        img_pipeline = make_pipeline({"sexual_explicit": 0.02})
        prompt_pipeline_mock = MagicMock()
        prompt_pipeline_mock.check.return_value = PromptGuardResult(
            model="mock", verdict="BLOCK", category="sexual_explicit",
            confidence=0.95, rationale="Bad prompt."
        )

        combined = CombinedPipeline(
            image_pipeline=img_pipeline,
            prompt_pipeline=prompt_pipeline_mock,
        )
        verdict = combined.run(image_path=safe_image_path, prompt="Generate explicit content")
        assert verdict.decision == "BLOCK"
        assert verdict.prompt_verdict == "BLOCK"

    def test_prompt_allow_proceeds_to_image(self, safe_image_path):
        img_pipeline = make_pipeline({"sexual_explicit": 0.02})
        prompt_pipeline_mock = MagicMock()
        prompt_pipeline_mock.check.return_value = PromptGuardResult(
            model="mock", verdict="ALLOW", category=None,
            confidence=0.97, rationale="Safe prompt."
        )

        combined = CombinedPipeline(
            image_pipeline=img_pipeline,
            prompt_pipeline=prompt_pipeline_mock,
        )
        verdict = combined.run(image_path=safe_image_path, prompt="Beach scene")
        assert verdict.decision == "ALLOW"
        assert verdict.prompt_verdict == "ALLOW"

    def test_prompt_only_allow(self):
        img_pipeline = make_pipeline({"sexual_explicit": 0.02})
        prompt_pipeline_mock = MagicMock()
        prompt_pipeline_mock.check.return_value = PromptGuardResult(
            model="mock", verdict="ALLOW", category=None,
            confidence=0.97, rationale="Safe."
        )

        combined = CombinedPipeline(
            image_pipeline=img_pipeline,
            prompt_pipeline=prompt_pipeline_mock,
        )
        verdict = combined.run(prompt="A scenic landscape")
        assert verdict.decision == "ALLOW"

    def test_no_input_raises(self, safe_image_path):
        img_pipeline = make_pipeline({"sexual_explicit": 0.02})
        combined = CombinedPipeline(image_pipeline=img_pipeline)
        with pytest.raises(ValueError):
            combined.run()


# ---------------------------------------------------------------------------
# Tests: Batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_batch_returns_correct_count(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.02})
        images = [safe_image_path, safe_image_path, safe_image_path]
        verdicts = pipeline.run_batch(images)
        assert len(verdicts) == 3

    def test_batch_all_allow(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.02})
        verdicts = pipeline.run_batch([safe_image_path] * 5)
        assert all(v.decision == "ALLOW" for v in verdicts)


# ---------------------------------------------------------------------------
# Tests: Verdict fields and metadata
# ---------------------------------------------------------------------------

class TestVerdictFields:
    def test_verdict_has_image_id(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.02})
        verdict = pipeline.run(safe_image_path)
        assert verdict.image_id != ""
        assert len(verdict.image_id) == 64  # SHA-256 hex

    def test_verdict_has_timestamp(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.02})
        verdict = pipeline.run(safe_image_path)
        assert "T" in verdict.timestamp  # ISO-8601

    def test_verdict_has_latency(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.02})
        verdict = pipeline.run(safe_image_path)
        assert verdict.latency_ms > 0

    def test_verdict_has_explanation(self, safe_image_path):
        pipeline = make_pipeline({"sexual_explicit": 0.95})
        verdict = pipeline.run(safe_image_path)
        assert verdict.explanation_for_user
        assert verdict.explanation_for_regulator


# ---------------------------------------------------------------------------
# Tests: NaN guard — float16 overflow must not silently cause REVIEW/BLOCK
# ---------------------------------------------------------------------------

class TestNanGuard:
    """NaN scores arise when ShieldGemma-2 is loaded with float16 instead of
    bfloat16 on MPS/Apple Silicon.  The pipeline must intercept NaN and treat
    it as a safe (ALLOW) result rather than silently falling into the gray zone
    and invoking the reasoner with garbage data."""

    def test_nan_scores_are_treated_as_allow(self, safe_image_path):
        nan = float("nan")
        pipeline = make_pipeline({
            "sexual_explicit": nan,
            "violence_gore":   nan,
            "extremism":       nan,
        })
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "ALLOW", (
            "NaN classifier scores must be treated as ALLOW, not trigger the reasoner"
        )

    def test_nan_scores_do_not_call_reasoner(self, safe_image_path):
        nan = float("nan")
        rsn = MockReasoner("BLOCK", "extremism", 0.99)
        clf = MockClassifier(scores={
            "sexual_explicit": nan,
            "violence_gore":   nan,
            "extremism":       nan,
        })
        pipeline = ImagePipeline(
            classifier=clf,
            reasoner=rsn,
            taxonomy=TaxonomyLoader(TAXONOMY_PATH),
            policy_engine=PolicyEngine(TaxonomyLoader(TAXONOMY_PATH), POLICY_PATH),
        )
        verdict = pipeline.run(safe_image_path)
        assert rsn.call_count == 0, "Reasoner must NOT be called when all scores are NaN"
        assert verdict.decision == "ALLOW"

    def test_nan_scores_replaced_in_verdict(self, safe_image_path):
        nan = float("nan")
        pipeline = make_pipeline({
            "sexual_explicit": nan,
            "violence_gore":   nan,
            "extremism":       nan,
        })
        verdict = pipeline.run(safe_image_path)
        import math
        for cat, score in verdict.classifier_scores.items():
            assert not math.isnan(score), f"NaN found in verdict.classifier_scores[{cat!r}]"

    def test_mixed_nan_and_valid_scores_use_valid_max(self, safe_image_path):
        """If one category is NaN but another is a valid high score, that score is used."""
        nan = float("nan")
        pipeline = make_pipeline({
            "sexual_explicit": 0.95,
            "violence_gore":   nan,
            "extremism":       nan,
        })
        verdict = pipeline.run(safe_image_path)
        assert verdict.decision == "BLOCK"


# ---------------------------------------------------------------------------
# Tests: Classifier dtype & API correctness (structural)
# ---------------------------------------------------------------------------

class TestClassifierStructure:
    """Verify the ShieldGemma2Classifier is wired with bfloat16 by default
    and uses the correct ShieldGemma2Processor policies= API."""

    def test_shieldgemma2_default_dtype_is_bfloat16(self):
        from censorship.classifiers.shieldgemma2 import ShieldGemma2Classifier
        clf = ShieldGemma2Classifier()
        assert clf.torch_dtype == "bfloat16", (
            "Default dtype must be bfloat16 — float16 causes NaN on MPS"
        )

    def test_shieldgemma2_policy_key_map_covers_all_categories(self):
        from censorship.classifiers.shieldgemma2 import ShieldGemma2Classifier, _POLICY_KEY_MAP
        clf = ShieldGemma2Classifier()
        for cat in clf.supported_categories:
            assert cat in _POLICY_KEY_MAP, (
                f"Category '{cat}' missing from _POLICY_KEY_MAP — every supported "
                "category needs a built-in ShieldGemma2Processor policy key"
            )

    def test_shieldgemma2_reasoner_default_dtype_is_bfloat16(self):
        from censorship.reasoners.shieldgemma2_reason import ShieldGemma2Reasoner
        rsn = ShieldGemma2Reasoner()
        assert rsn.torch_dtype == "bfloat16"

    def test_registry_shieldgemma2_dtype_from_config(self):
        from censorship.classifiers.registry import get_classifier
        clf = get_classifier("shieldgemma2", {"torch_dtype": "bfloat16"})
        assert clf.torch_dtype == "bfloat16"

    def test_registry_shieldgemma2_default_dtype_is_bfloat16(self):
        from censorship.classifiers.registry import get_classifier
        clf = get_classifier("shieldgemma2", {})
        assert clf.torch_dtype == "bfloat16"

    def test_device_utils_mps_returns_bfloat16(self):
        """auto_device_dtype must return bfloat16 (not float16) on MPS."""
        import importlib, unittest.mock
        device_utils = importlib.import_module("censorship.core.device_utils")
        import torch
        with unittest.mock.patch.object(
            torch.backends.mps, "is_available", return_value=True
        ):
            device, dtype = device_utils.auto_device_dtype("float16")
        assert dtype == torch.bfloat16, (
            "MPS must use bfloat16 — float16 causes NaN in PaliGemma-2 attention"
        )
        assert device == "mps"
