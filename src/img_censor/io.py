from pathlib import Path
from typing import Tuple

import requests
from PIL import Image


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    return Image.open(Path(path_or_url)).convert("RGB")


def select_device_and_dtype(runtime_config: dict) -> Tuple[str, object]:
    import torch

    configured_device = runtime_config.get("device", "auto")
    configured_dtype = runtime_config.get("dtype", "auto")

    if configured_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = configured_device

    if configured_dtype == "auto":
        if device in {"cuda", "mps"}:
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, configured_dtype)

    return device, dtype


def move_batch_to_device(batch: dict, device: str, dtype: object) -> dict:
    import torch

    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved

