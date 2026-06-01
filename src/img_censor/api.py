import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile

from img_censor.config import load_config
from img_censor.hackathon_service import HackathonCensorService
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
service = HackathonCensorService(pipeline)


@app.get("/")
async def root():
    return {
        "service": "Image Censorship Module",
        "docs": "/docs",
        "health": "/health",
        "censor": "/v1/censor",
        "prompt_censor": "/v1/censor/prompt",
        "input_image_censor": "/v1/censor/input-image",
        "output_image_censor": "/v1/censor/output-image",
        "full_flow": "/v1/censor/full",
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


@app.post("/v1/censor/prompt")
async def censor_prompt(prompt: str = Form(...), request_id: Optional[str] = Form(default=None)):
    request = GuardRequest(prompt=prompt, request_id=request_id)
    result = mock_check(request) if USE_MOCK else service.check_prompt(prompt, request_id=request_id)
    return result.to_dict()


@app.post("/v1/censor/input-image")
async def censor_input_image(
    input_image: UploadFile = File(...),
    request_id: Optional[str] = Form(default=None),
):
    input_path = await _save_upload(input_image)
    result = service.check_input_image(input_path, request_id=request_id)
    return result.to_dict()


@app.post("/v1/censor/output-image")
async def censor_output_image(
    output_image: UploadFile = File(...),
    request_id: Optional[str] = Form(default=None),
):
    output_path = await _save_upload(output_image)
    result = service.check_output_image(output_path, request_id=request_id)
    return result.to_dict()


@app.post("/v1/censor/full")
async def censor_full(
    prompt: Optional[str] = Form(default=None),
    input_image: Optional[UploadFile] = File(default=None),
    generated_image: Optional[UploadFile] = File(default=None),
    request_id: Optional[str] = Form(default=None),
    use_mock_generator: bool = Form(default=True),
):
    input_path = await _save_upload(input_image) if input_image else None
    generated_path = await _save_upload(generated_image) if generated_image else None
    if USE_MOCK:
        request = GuardRequest(prompt=prompt, input_image=input_path, output_image=generated_path, request_id=request_id)
        return mock_check(request).to_dict()
    return service.full_flow(
        prompt=prompt,
        input_image=input_path,
        generated_image=generated_path,
        request_id=request_id,
        use_mock_generator=use_mock_generator,
    )


async def _save_upload(upload: UploadFile) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.rsplit(".", 1)[-1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await upload.read())
        return handle.name
