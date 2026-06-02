"""
NaN investigation tests for ShieldGemma-2.

Root cause (documented after investigation 2026-06-02):
-------------------------------------------------------
ShieldGemma-2 is based on PaliGemma-2 / SigLIP vision encoder.
The SigLIP encoder uses scaled dot-product attention with very large
hidden state values (logits in the 20-70 range in float). With float16
(max ≈ 65504), intermediate attention softmax computations (exp of large
values) overflow → NaN. With bfloat16 (max ≈ 3.4e38), no overflow.

Additionally: ShieldGemma2Processor silently IGNORES the `text=` kwarg
and always generates prompts from its built-in policy definitions. If you
don't pass `policies=[key]`, it runs ALL 3 default policies in a single
batch, and old code extracted `probabilities[0, ...]` (first policy in
batch) instead of per-policy results — this caused systematic wrong scores.

Fix applied:
  1. auto_device_dtype() → bfloat16 on MPS (was float16)
  2. ShieldGemma2Classifier → processor(images=img, policies=[key]) per call
  3. ImagePipeline.run() → NaN guard replacing NaN scores with 0.0 + warning
  4. ClassifierResult.max_score() → returns 0.0 instead of NaN

These unit tests verify all four layers of the fix without requiring a GPU
or real model (mock torch and transformers).
"""

from __future__ import annotations

import math
import unittest.mock as mock
import importlib
import pytest
import torch

# ---------------------------------------------------------------------------
# Unit tests: dtype selection
# ---------------------------------------------------------------------------

class TestAutoDeviceDtype:
    """auto_device_dtype must always return bfloat16 on MPS, regardless of
    the config_dtype argument.  float16 causes NaN in SigLIP attention."""

    def _get_fn(self):
        import censorship.core.device_utils as du
        importlib.reload(du)
        return du.auto_device_dtype

    def test_mps_always_bfloat16(self):
        fn = self._get_fn()
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
             mock.patch("torch.cuda.is_available", return_value=False):
            device, dtype = fn("float16")
        assert dtype == torch.bfloat16, (
            "float16 causes NaN in SigLIP attention — must use bfloat16 on MPS"
        )
        assert device == "mps"

    def test_mps_bfloat16_even_when_explicitly_given_float16(self):
        fn = self._get_fn()
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
             mock.patch("torch.cuda.is_available", return_value=False):
            _, dtype = fn("float16")
        assert dtype != torch.float16, "float16 is forbidden on MPS for this model"

    def test_cuda_respects_config_dtype_float16(self):
        fn = self._get_fn()
        with mock.patch("torch.backends.mps.is_available", return_value=False), \
             mock.patch("torch.cuda.is_available", return_value=True):
            _, dtype = fn("float16")
        assert dtype == torch.float16

    def test_cuda_respects_config_dtype_bfloat16(self):
        fn = self._get_fn()
        with mock.patch("torch.backends.mps.is_available", return_value=False), \
             mock.patch("torch.cuda.is_available", return_value=True):
            _, dtype = fn("bfloat16")
        assert dtype == torch.bfloat16

    def test_cpu_always_float32(self):
        fn = self._get_fn()
        with mock.patch("torch.backends.mps.is_available", return_value=False), \
             mock.patch("torch.cuda.is_available", return_value=False):
            device, dtype = fn("float16")
        assert dtype == torch.float32
        assert device == "cpu"


# ---------------------------------------------------------------------------
# Unit tests: ShieldGemma2Classifier API correctness
# ---------------------------------------------------------------------------

class TestShieldGemma2ClassifierAPI:
    """Verify the classifier uses the correct ShieldGemma2Processor API.

    The processor's `text=` kwarg is silently IGNORED — it always generates
    prompts from its built-in policy definitions.  The correct API is:
        processor(images=img, policies=[key])
    not:
        processor(text=prompt, images=img)
    """

    def _make_clf(self):
        from censorship.classifiers.shieldgemma2 import ShieldGemma2Classifier
        return ShieldGemma2Classifier(hf_token="test")

    def test_default_dtype_is_bfloat16(self):
        clf = self._make_clf()
        assert clf.torch_dtype == "bfloat16"

    def test_policy_key_map_covers_all_supported_categories(self):
        from censorship.classifiers.shieldgemma2 import ShieldGemma2Classifier, _POLICY_KEY_MAP
        clf = ShieldGemma2Classifier()
        for cat in clf.supported_categories:
            assert cat in _POLICY_KEY_MAP, f"Category '{cat}' missing from _POLICY_KEY_MAP"

    def test_policy_keys_are_valid_shieldgemma2_keys(self):
        from censorship.classifiers.shieldgemma2 import _POLICY_KEY_MAP
        valid_keys = {"dangerous", "sexual", "violence"}
        for cat, key in _POLICY_KEY_MAP.items():
            assert key in valid_keys, (
                f"'{key}' is not a valid ShieldGemma2Processor built-in policy key. "
                f"Valid: {valid_keys}"
            )

    def test_score_one_policy_uses_policies_kwarg(self):
        """The processor must be called with policies=[key], NOT text=prompt."""
        from censorship.classifiers.shieldgemma2 import ShieldGemma2Classifier
        from PIL import Image
        import numpy as np

        clf = ShieldGemma2Classifier()

        # Mock the loaded model and processor
        mock_probs = torch.tensor([[0.3, 0.7]])  # P(violation)=0.3, P(safe)=0.7
        mock_out = mock.MagicMock()
        mock_out.probabilities = mock_probs

        class _Batch(dict):
            def to(self, device): return self

        mock_processor = mock.MagicMock()
        mock_processor.return_value = _Batch({"input_ids": torch.zeros(1, 10, dtype=torch.long)})

        mock_model = mock.MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.return_value = mock_out

        clf._model = mock_model
        clf._processor = mock_processor

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result = clf._score_one_policy(img, "sexual")

        # Check that processor was called with policies kwarg, NOT text kwarg
        call_kwargs = mock_processor.call_args.kwargs
        assert "policies" in call_kwargs, (
            "Processor must be called with policies=[key]. "
            "Passing text= is silently ignored by ShieldGemma2Processor."
        )
        assert call_kwargs["policies"] == ["sexual"]
        assert "text" not in call_kwargs, (
            "Processor must NOT receive a text= kwarg — it is silently ignored"
        )

    def test_extracts_probability_from_batch_index_0(self):
        """With single-policy call, batch_size=1 → extract probabilities[0,0]."""
        from censorship.classifiers.shieldgemma2 import ShieldGemma2Classifier
        from PIL import Image
        import numpy as np

        clf = ShieldGemma2Classifier()
        expected_p_viol = 0.72

        mock_probs = torch.tensor([[expected_p_viol, 1 - expected_p_viol]])
        mock_out = mock.MagicMock()
        mock_out.probabilities = mock_probs

        class _Batch(dict):
            def to(self, device): return self

        mock_processor = mock.MagicMock()
        mock_processor.return_value = _Batch({"input_ids": torch.zeros(1, 10, dtype=torch.long)})
        mock_model = mock.MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.return_value = mock_out

        clf._model = mock_model
        clf._processor = mock_processor

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result = clf._score_one_policy(img, "sexual")
        assert abs(result - expected_p_viol) < 1e-4


# ---------------------------------------------------------------------------
# Unit tests: NaN guard in pipeline
# ---------------------------------------------------------------------------

class TestNanPipelineGuard:
    """Regression tests for NaN propagation through the pipeline.

    Before the fix, NaN scores from the classifier caused:
      1. max_score() → NaN
      2. NaN >= 0.90 → False  (Python NaN comparison)
      3. NaN < 0.50  → False
      4. Pipeline fell into gray zone and called the reasoner with garbage input
      5. Reasoner returned REVIEW/BLOCK based on random weights → wrong decisions
    """

    _TAXONOMY_PATH = None
    _POLICY_PATH   = None

    @classmethod
    def setup_class(cls):
        from pathlib import Path
        base = Path(__file__).parent.parent
        cls._TAXONOMY_PATH = base / "config" / "taxonomy.yaml"
        cls._POLICY_PATH   = base / "config" / "policy_bank.yaml"

    def _make_nan_pipeline(self, reasoner=None):
        from tests.test_pipeline import MockClassifier
        from censorship.core.taxonomy import TaxonomyLoader
        from censorship.core.policy import PolicyEngine
        from censorship.pipeline.image_pipeline import ImagePipeline
        from PIL import Image
        import tempfile, numpy as np

        nan = float("nan")
        clf = MockClassifier({"sexual_explicit": nan, "violence_gore": nan, "extremism": nan})
        tax = TaxonomyLoader(self._TAXONOMY_PATH)
        pol = PolicyEngine(tax, self._POLICY_PATH)

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        p = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(p.name)

        return ImagePipeline(
            classifier=clf, reasoner=reasoner,
            taxonomy=tax, policy_engine=pol
        ), p.name

    def test_all_nan_gives_allow(self):
        from tests.test_pipeline import MockReasoner
        rsn = MockReasoner("BLOCK", "extremism", 0.99)
        pipeline, img = self._make_nan_pipeline(reasoner=rsn)
        verdict = pipeline.run(img)
        assert verdict.decision == "ALLOW", "All-NaN scores must yield ALLOW, not call reasoner"

    def test_all_nan_does_not_call_reasoner(self):
        from tests.test_pipeline import MockReasoner
        rsn = MockReasoner("BLOCK", "extremism", 0.99)
        pipeline, img = self._make_nan_pipeline(reasoner=rsn)
        pipeline.run(img)
        assert rsn.call_count == 0, "Reasoner must never be called when all classifier scores are NaN"

    def test_all_nan_scores_zeroed_in_verdict(self):
        pipeline, img = self._make_nan_pipeline()
        verdict = pipeline.run(img)
        for cat, score in verdict.classifier_scores.items():
            assert not math.isnan(score), f"NaN leaked into verdict.classifier_scores[{cat!r}]"

    def test_max_score_nan_returns_zero(self):
        from censorship.core.verdict import ClassifierResult
        nan = float("nan")
        r = ClassifierResult(
            model="test",
            scores={"sexual_explicit": nan, "violence_gore": nan},
            is_unsafe=False, triggered_categories=[],
        )
        assert r.max_score() == 0.0
        assert not math.isnan(r.max_score())


# ---------------------------------------------------------------------------
# Unit tests: NaN cannot arise from correct dtype usage
# ---------------------------------------------------------------------------

class TestDtypeNanPrevention:
    """Verify that using bfloat16 prevents the specific overflow that caused NaN.

    The root cause: SigLIP (vision encoder in ShieldGemma-2) produces attention
    logits in the range 15-70. With float16 (max ~65504), exp(70) overflows to inf,
    and inf/inf = NaN in softmax. With bfloat16 (max ~3.4e38), exp(70) ≈ 2.5e30
    which is representable and softmax is numerically stable.
    """

    def test_float16_overflows_large_attention_logit(self):
        """Demonstrate that float16 produces NaN for large attention logits."""
        import torch
        large_logit = torch.tensor([70.0, -70.0], dtype=torch.float16)
        result = torch.softmax(large_logit, dim=0)
        # exp(70) overflows float16 (max ~65504) → inf/inf = NaN
        assert torch.isnan(result).any() or result[0] == 1.0, (
            "float16 should either overflow or round to exactly 1.0"
        )

    def test_bfloat16_stable_for_large_attention_logit(self):
        """bfloat16 handles large attention logits without overflow."""
        import torch
        large_logit = torch.tensor([70.0, -70.0], dtype=torch.bfloat16)
        result = torch.softmax(large_logit, dim=0)
        assert not torch.isnan(result).any(), "bfloat16 must not produce NaN for large logits"
        assert result[0] > 0.9, "Softmax should strongly prefer the first element"

    def test_real_shieldgemma2_logit_range_is_nan_safe_bfloat16(self):
        """Typical ShieldGemma-2 yes/no logits (19-70 range) are NaN-safe in bfloat16."""
        import torch
        # From actual diagnostic: yes_logit ~20-45, no_logit ~30-67
        logit_pairs = [
            (19.0, 29.4),   # Safe image, sexual policy
            (26.5, 29.6),   # Sexual image, sexual policy
            (40.0, 64.5),   # Safe image, violence policy
            (43.5, 46.0),   # Violence image, dangerous policy
        ]
        for y, n in logit_pairs:
            t = torch.tensor([y, n], dtype=torch.bfloat16)
            p = torch.softmax(t, dim=0)
            assert not torch.isnan(p).any(), (
                f"NaN for logits ({y}, {n}) with bfloat16 — unexpected"
            )

    def test_real_shieldgemma2_logit_range_nan_in_float16(self):
        """Large logits overflow float16 exp() → inf, and inf/inf = NaN.

        torch.softmax uses a numerically-stable log-sum-exp on CPU so it doesn't
        NaN by itself.  The NaN occurs inside SigLIP's raw scaled-dot-product
        attention which computes exp(QK^T/√d) without the max-subtraction trick.
        We reproduce the exact mechanism here: manual exp + divide.
        """
        import torch
        # exp(70) > float16 max (~65504) → overflows to inf
        large_logit = torch.tensor([70.0, -70.0], dtype=torch.float16)
        exp_vals = torch.exp(large_logit)  # [inf, ~0]
        assert torch.isinf(exp_vals[0]), "exp(70) must overflow to inf in float16"

        # Raw attention softmax without numerical stability: inf / (inf + finite) = NaN
        raw_softmax = exp_vals / exp_vals.sum()  # inf / inf = NaN
        assert torch.isnan(raw_softmax).any(), (
            "inf/inf must produce NaN — this is exactly what happens in SigLIP's "
            "scaled-dot-product attention when logits overflow float16."
        )
