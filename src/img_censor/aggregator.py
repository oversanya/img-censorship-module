from typing import Dict, List

from img_censor.schemas import Finding, GuardResult, Verdict


VERDICT_ORDER = {
    Verdict.ALLOW: 0,
    Verdict.REVIEW: 1,
    Verdict.BLOCK: 2,
}


def aggregate(findings: List[Finding], decision_config: Dict) -> GuardResult:
    fail_closed = set(decision_config.get("fail_closed_for", []))

    categories = sorted({finding.category for finding in findings if finding.category != "none"})
    if not findings:
        return GuardResult(
            verdict=Verdict.ALLOW,
            categories=[],
            rationale="No detector reported a policy violation above review threshold.",
            findings=[],
            audit={"decision_rule": "no_findings"},
        )

    final_verdict = max((finding.verdict for finding in findings), key=lambda value: VERDICT_ORDER[value])
    if any(finding.category in fail_closed for finding in findings):
        final_verdict = Verdict.BLOCK

    if final_verdict == Verdict.BLOCK:
        rationale = "Blocked because at least one detector reported a high-confidence or fail-closed violation."
    elif final_verdict == Verdict.REVIEW:
        rationale = "Sent to manual review because one or more detectors found medium-confidence risk."
    else:
        rationale = "Allowed because no finding exceeded the configured thresholds."

    return GuardResult(
        verdict=final_verdict,
        categories=categories,
        rationale=rationale,
        findings=findings,
        audit={
            "decision_rule": "max_detector_verdict_with_fail_closed_override",
            "fail_closed_for": sorted(fail_closed),
            "finding_count": len(findings),
        },
    )

