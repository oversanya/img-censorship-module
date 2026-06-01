from img_censor.schemas import Finding, GuardRequest, GuardResult, Verdict


def mock_check(request: GuardRequest) -> GuardResult:
    findings = []
    text = (request.prompt or "").lower()
    if "bomb" in text or "бомб" in text:
        findings.append(
            Finding(
                detector="mock",
                stage="prompt",
                category="dangerous",
                score=0.99,
                verdict=Verdict.BLOCK,
                rationale="Mock detector matched dangerous prompt intent.",
                raw={},
            )
        )

    if findings:
        return GuardResult(
            verdict=Verdict.BLOCK,
            categories=sorted({finding.category for finding in findings}),
            rationale="Mock pipeline blocked the request.",
            findings=findings,
            audit={"mock": True},
        )

    return GuardResult(
        verdict=Verdict.ALLOW,
        categories=[],
        rationale="Mock pipeline found no violation.",
        findings=[],
        audit={"mock": True},
    )

