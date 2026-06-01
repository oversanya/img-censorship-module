import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile

from img_censor.config import load_config
from img_censor.mock import mock_check
from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import GuardRequest


app = FastAPI(title="Image Censorship Module", version="0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"
CONFIG_PATH = Path(os.environ.get("IMG_CENSOR_CONFIG", DEFAULT_CONFIG_PATH))
USE_MOCK = os.environ.get("IMG_CENSOR_MOCK", "").lower() in {"1", "true", "yes"}

config = load_config(str(CONFIG_PATH))
pipeline = ImageCensorPipeline(config)


@app.get("/")
async def root():
    return {
        "service": "Image Censorship Module",
        "docs": "/docs",
        "health": "/health",
        "censor": "/v1/censor",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "config": str(CONFIG_PATH),
        "mock": USE_MOCK,
        "detectors": pipeline.describe()["detectors"],
    }


@app.post("/v1/censor")
async def censor(
    prompt: Optional[str] = Form(default=None),
    input_image: Optional[UploadFile] = File(default=None),
    output_image: Optional[UploadFile] = File(default=None),
):
    input_path = await _save_upload(input_image) if input_image else None
    output_path = await _save_upload(output_image) if output_image else None
    request = GuardRequest(
        prompt=prompt,
        input_image=input_path,
        output_image=output_path,
    )
    result = mock_check(request) if USE_MOCK else pipeline.check(request)
    return result.to_dict()


async def _save_upload(upload: UploadFile) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.rsplit(".", 1)[-1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await upload.read())
        return handle.name
