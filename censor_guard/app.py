from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest, ModerationResponse


app = FastAPI(
    title="Censor Module MVP",
    description="Multi-layer guardrail prototype for multimodal image generation.",
    version="0.1.0",
)

pipeline = GuardrailPipeline()
ui_dir = Path(__file__).resolve().parent.parent / "ui"

if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest) -> ModerationResponse:
    return pipeline.moderate(request)


@app.get("/", response_class=FileResponse)
def frontend() -> FileResponse:
    return FileResponse(ui_dir / "index.html")

