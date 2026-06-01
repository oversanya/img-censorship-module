from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile

from img_censor.config import load_config
from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import GuardRequest


app = FastAPI(title="Image Censorship Module", version="0.1.0")
pipeline = ImageCensorPipeline(load_config("configs/pipeline.yaml"))


@app.post("/v1/censor")
async def censor(
    prompt: Optional[str] = Form(default=None),
    input_image: Optional[UploadFile] = File(default=None),
    output_image: Optional[UploadFile] = File(default=None),
):
    input_path = await _save_upload(input_image) if input_image else None
    output_path = await _save_upload(output_image) if output_image else None
    result = pipeline.check(
        GuardRequest(
            prompt=prompt,
            input_image=input_path,
            output_image=output_path,
        )
    )
    return result.to_dict()


async def _save_upload(upload: UploadFile) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.rsplit(".", 1)[-1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await upload.read())
        return handle.name

