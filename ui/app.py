from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from censor_guard.observability import ObservabilityError
from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest, ModerationResponse


app = FastAPI(
    title="Censor Module UI Prototype",
    description="UI prototype for the multimodal image moderation guardrail.",
    version="0.1.0",
)

pipeline = GuardrailPipeline()
static_dir = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.exception_handler(ObservabilityError)
def observability_error_handler(_, exc: ObservabilityError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "moderation logging failed", "error": str(exc)},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest) -> ModerationResponse:
    return pipeline.moderate(request)


@app.get("/", response_class=FileResponse)
def frontend() -> FileResponse:
    return FileResponse(static_dir / "index.html")
