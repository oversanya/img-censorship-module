"""Tests for core verdict dataclasses."""

import json
import math
import pytest
from censorship.core.verdict import (
    Verdict,
    ClassifierResult,
    ReasonerResult,
    PromptGuardResult,
    _now_iso,
)


def test_classifier_result_max_score():
    result = ClassifierResult(
        model="test",
        scores={"sexual_explicit": 0.3, "violence_gore": 0.7, "extremism": 0.1},
        is_unsafe=True,
        triggered_categories=["violence_gore"],
    )
    assert result.max_score() == pytest.approx(0.7)
    assert result.top_category() == "violence_gore"


def test_classifier_result_empty_scores():
    result = ClassifierResult(
        model="test",
        scores={},
        is_unsafe=False,
        triggered_categories=[],
    )
    assert result.max_score() == 0.0
    assert result.top_category() is None


def test_classifier_result_nan_scores_become_zero():
    """NaN scores must be coerced to 0.0 so the pipeline never silently falls into
    the gray-zone due to float16 overflow (a real bug we fixed in ShieldGemma-2)."""
    nan = float("nan")
    result = ClassifierResult(
        model="test",
        scores={"sexual_explicit": nan, "violence_gore": nan, "extremism": nan},
        is_unsafe=False,
        triggered_categories=[],
    )
    assert result.max_score() == 0.0
    assert not math.isnan(result.max_score())


def test_reasoner_result_fields():
    r = ReasonerResult(
        model="llavaguard",
        verdict="BLOCK",
        category="sexual_explicit",
        confidence=0.93,
        rationale="Explicit content detected.",
    )
    assert r.verdict == "BLOCK"
    assert r.confidence == pytest.approx(0.93)


def test_prompt_guard_result():
    r = PromptGuardResult(
        model="llamaguard4",
        verdict="ALLOW",
        category=None,
        confidence=0.97,
    )
    assert r.verdict == "ALLOW"
    assert r.category is None


def test_verdict_serialization():
    v = Verdict(
        image_id="abc123",
        timestamp=_now_iso(),
        decision="BLOCK",
        classifier_model="shieldgemma2",
        classifier_scores={"sexual_explicit": 0.95},
        classifier_triggered=["sexual_explicit"],
        primary_category="sexual_explicit",
        explanation_for_user="Blocked.",
        explanation_for_regulator="Blocked by policy.",
        latency_ms=150.5,
    )
    d = v.to_dict()
    assert d["decision"] == "BLOCK"
    assert d["image_id"] == "abc123"
    assert d["classifier_scores"]["sexual_explicit"] == pytest.approx(0.95)

    j = v.to_json()
    parsed = json.loads(j)
    assert parsed["decision"] == "BLOCK"


def test_verdict_from_classifier_allow():
    clf_result = ClassifierResult(
        model="shieldgemma2",
        scores={"sexual_explicit": 0.1, "violence_gore": 0.05},
        is_unsafe=False,
        triggered_categories=[],
    )
    v = Verdict.from_classifier_allow("hash123", clf_result, latency_ms=45.0)
    assert v.decision == "ALLOW"
    assert v.classifier_model == "shieldgemma2"
    assert v.latency_ms == pytest.approx(45.0)


def test_verdict_from_classifier_block():
    clf_result = ClassifierResult(
        model="shieldgemma2",
        scores={"sexual_explicit": 0.93, "violence_gore": 0.2},
        is_unsafe=True,
        triggered_categories=["sexual_explicit"],
    )
    v = Verdict.from_classifier_block("hash456", clf_result, {"sexual_explicit": 0.80})
    assert v.decision == "BLOCK"
    assert v.primary_category == "sexual_explicit"
