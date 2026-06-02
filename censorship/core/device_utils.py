"""Device and dtype auto-selection for Apple Silicon, CUDA, and CPU."""

from __future__ import annotations


def auto_device_dtype(config_dtype: str = "float16"):
    """
    Return (device_str, torch_dtype) appropriate for the current hardware.

    Priority: MPS (Apple Silicon) > CUDA > CPU
    bfloat16 is used on MPS — float16 causes NaN in ShieldGemma-2 attention.
    float32 is used on CPU for numerical stability.
    """
    import torch

    if torch.backends.mps.is_available():
        # Apple Silicon: bfloat16 required — float16 causes NaN in PaliGemma-2 attention
        dtype = torch.bfloat16
        device = "mps"
    elif torch.cuda.is_available():
        dtype = getattr(torch, config_dtype, torch.float16)
        device = "cuda"
    else:
        # CPU: float32 avoids precision issues in pure Python execution
        dtype = torch.float32
        device = "cpu"

    return device, dtype


def load_to_device(model, device: str):
    """Move model to device.

    Skips only when device_map='auto' was used (hf_device_map is a non-empty dict),
    because in that case the model has already been sharded across devices.
    """
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        # Model was loaded with device_map='auto' — already placed, skip
        return model
    return model.to(device)
