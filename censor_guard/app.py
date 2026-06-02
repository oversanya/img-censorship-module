from __future__ import annotations

from fastapi import FastAPI

from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest, ModerationResponse


app = FastAPI(
    title="Censor Module MVP",
    description="Multi-layer guardrail prototype for multimodal image generation.",
    version="0.1.0",
)

pipeline = GuardrailPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest) -> ModerationResponse:
    return pipeline.moderate(request)

