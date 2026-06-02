"""Tests for PolicyEngine threshold logic."""

import pytest
from pathlib import Path

from censorship.core.taxonomy import TaxonomyLoader
from censorship.core.policy import PolicyEngine, Decision


TAXONOMY_PATH = Path(__file__).parent.parent / "config" / "taxonomy.yaml"
POLICY_PATH = Path(__file__).parent.parent / "config" / "policy_bank.yaml"


@pytest.fixture
def taxonomy():
    return TaxonomyLoader(TAXONOMY_PATH)


@pytest.fixture
def engine(taxonomy):
    return PolicyEngine(taxonomy, POLICY_PATH)


def test_allow_low_scores(engine):
    # All scores below per-category review thresholds (sexual=0.10, violence/extremism=0.05)
    scores = {"sexual_explicit": 0.05, "violence_gore": 0.02, "extremism": 0.01}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.ALLOW
    assert dec.category is None


def test_block_high_score(engine):
    scores = {"sexual_explicit": 0.95, "violence_gore": 0.1}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.BLOCK
    assert dec.category == "sexual_explicit"
    assert dec.confidence == pytest.approx(0.95)


def test_review_middle_score(engine):
    # 0.60 is in review zone for sexual_explicit (block=0.80, review=0.10)
    scores = {"sexual_explicit": 0.60, "violence_gore": 0.02}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.REVIEW
    assert dec.category == "sexual_explicit"


def test_csam_is_critical_block(engine):
    # CSAM is critical — any score > 0 must block
    scores = {"csam": 0.01}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.BLOCK
    assert dec.category == "csam"


def test_csam_always_blocks_even_tiny_score(engine):
    scores = {"csam": 0.001, "sexual_explicit": 0.10}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.BLOCK
    assert dec.category == "csam"


def test_block_threshold_boundary(engine):
    # Exactly at block threshold for sexual_explicit (0.80)
    scores = {"sexual_explicit": 0.80}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.BLOCK


def test_below_review_threshold(engine):
    # Just below review threshold for sexual_explicit (0.10)
    scores = {"sexual_explicit": 0.05}
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.ALLOW


def test_needs_reasoner_gray_zone(engine):
    # In gray zone: between fast_allow (0.05) and fast_block (0.80)
    assert engine.needs_reasoner(0.30) is True
    assert engine.needs_reasoner(0.05) is True


def test_needs_reasoner_outside_zone(engine):
    assert engine.needs_reasoner(0.95) is False  # above fast_block → no reasoner needed
    assert engine.needs_reasoner(0.03) is False  # below fast_allow → allow directly


def test_highest_score_category_blocks(engine):
    # When multiple categories trigger, the highest-confidence one is reported
    scores = {
        "sexual_explicit": 0.82,
        "violence_gore": 0.91,  # highest, above block threshold
        "extremism": 0.60,
    }
    dec = engine.evaluate(scores)
    assert dec.decision == Decision.BLOCK
    assert dec.category == "violence_gore"


def test_taxonomy_ids_loaded(taxonomy):
    ids = taxonomy.ids()
    assert "sexual_explicit" in ids
    assert "csam" in ids
    assert "violence_gore" in ids
    assert len(ids) == 7


def test_taxonomy_critical_category(taxonomy):
    assert taxonomy.is_critical("csam") is True
    assert taxonomy.is_critical("sexual_explicit") is False
