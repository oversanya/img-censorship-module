from typing import List

from img_censor.detectors.base import Detector
from img_censor.policy import PROMPT_KEYWORDS
from img_censor.schemas import Finding, GuardRequest, Verdict
from img_censor.text_normalization import compact_text, normalize_text_variants


class PromptKeywordGuard(Detector):
    name = "prompt_keywords"

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled or not request.prompt:
            return []

        prompt_variants = normalize_text_variants(request.prompt)
        findings = []
        for category, keywords in PROMPT_KEYWORDS.items():
            matched = self._matched_keywords(prompt_variants, keywords)
            if matched:
                findings.append(
                    Finding(
                        detector=self.name,
                        stage="prompt",
                        category=category,
                        score=1.0,
                        verdict=Verdict.BLOCK,
                        rationale="Prompt contains explicit high-risk keyword(s).",
                        raw={"matched_keywords": matched},
                    )
                )
        return findings

    @staticmethod
    def _matched_keywords(prompt_variants: List[str], keywords: List[str]) -> List[str]:
        matched = []
        for keyword in keywords:
            keyword_variants = normalize_text_variants(keyword)
            compact_keyword = compact_text(keyword)
            if any(keyword_variant in prompt_variants for keyword_variant in keyword_variants):
                matched.append(keyword)
            elif compact_keyword and any(compact_keyword in prompt_variant for prompt_variant in prompt_variants):
                matched.append(keyword)
        return matched
