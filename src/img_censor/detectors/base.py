from typing import List

from img_censor.schemas import Finding, GuardRequest


class Detector:
    name = "detector"

    def __init__(self, config: dict, runtime_config: dict):
        self.config = config
        self.runtime_config = runtime_config
        self.stages = set(config.get("stages", []))

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def run(self, request: GuardRequest) -> List[Finding]:
        raise NotImplementedError

    def should_run_on(self, stage: str) -> bool:
        return stage in self.stages

    def threshold_verdict(self, score: float):
        from img_censor.schemas import Verdict

        block_threshold = float(self.config.get("block_threshold", 0.70))
        review_threshold = float(self.config.get("review_threshold", 0.35))
        if score >= block_threshold:
            return Verdict.BLOCK
        if score >= review_threshold:
            return Verdict.REVIEW
        return Verdict.ALLOW

