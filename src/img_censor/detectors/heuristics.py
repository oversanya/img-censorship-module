from typing import List

from img_censor.detectors.base import Detector
from img_censor.policy import PROMPT_KEYWORDS
from img_censor.schemas import Finding, GuardRequest, Verdict


class PromptKeywordGuard(Detector):
    name = "prompt_keywords"

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled or not request.prompt:
            return []

        prompt = request.prompt.lower()
        findings = []
        for category, keywords in PROMPT_KEYWORDS.items():
            matched = [keyword for keyword in keywords if keyword in prompt]
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

