from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from censor_guard.schemas import ModerationRequest


def load_image(request: ModerationRequest) -> Image.Image | None:
    if request.image_base64:
        payload = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(payload))
        return image.convert("RGB")
    if request.image_path:
        image = Image.open(Path(request.image_path))
        return image.convert("RGB")
    return None

