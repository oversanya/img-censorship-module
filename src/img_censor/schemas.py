from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class GuardRequest:
    prompt: Optional[str] = None
    input_image: Optional[str] = None
    output_image: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class Finding:
    detector: str
    stage: str
    category: str
    score: float
    verdict: Verdict
    rationale: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "stage": self.stage,
            "category": self.category,
            "score": round(float(self.score), 6),
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "raw": self.raw,
        }


@dataclass
class GuardResult:
    verdict: Verdict
    categories: List[str]
    rationale: str
    findings: List[Finding]
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "categories": self.categories,
            "rationale": self.rationale,
            "findings": [finding.to_dict() for finding in self.findings],
            "audit": self.audit,
        }

