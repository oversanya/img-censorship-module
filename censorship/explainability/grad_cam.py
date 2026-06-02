"""GradCAM explainability for CNN-based classifiers (NudeNet, Q16).

Uses pytorch-grad-cam library to produce heatmaps showing which image
regions contributed to the classification decision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """
    Generates GradCAM heatmaps for CNN-based image classifiers.

    For ViT/CLIP models: uses GradCAM variant that works with attention blocks.
    For CNN models: uses standard GradCAM on the last conv layer.
    """

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model,
        image_path: Union[str, Path],
        target_layer,
        image_id: str,
        target_class: int = 1,
    ) -> Path | None:
        """
        Generate GradCAM heatmap and save overlay image.

        Args:
            model: PyTorch model
            image_path: path to input image
            target_layer: target layer for CAM (e.g., model.features[-1])
            image_id: used for output filename
            target_class: class index to explain

        Returns:
            Path to saved overlay image, or None on failure
        """
        try:
            import torch
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import (
                show_cam_on_image,
                preprocess_image,
            )
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image, dtype=np.float32) / 255.0

            input_tensor = preprocess_image(image_np).unsqueeze(0)

            targets = [ClassifierOutputTarget(target_class)]
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            overlay = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
            output_path = self.output_dir / f"{image_id}_gradcam.png"
            Image.fromarray(overlay).save(output_path)
            logger.info(f"GradCAM saved to {output_path}")
            return output_path

        except ImportError:
            logger.warning("pytorch-grad-cam not installed. Skipping GradCAM.")
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}")
        return None

    def generate_vit(
        self,
        model,
        processor,
        image_path: Union[str, Path],
        image_id: str,
    ) -> Path | None:
        """
        Generate attention-rollout visualization for ViT-based models (Q16).
        """
        try:
            import torch
            from pytorch_grad_cam import EigenCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image, dtype=np.float32) / 255.0

            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

            # For ViT: target last attention layer
            target_layer = model.vision_model.encoder.layers[-1].self_attn \
                if hasattr(model, "vision_model") else None

            if target_layer is None:
                logger.warning("Could not identify target layer for ViT GradCAM.")
                return None

            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1:, :].reshape(
                    tensor.size(0), height, width, tensor.size(2)
                )
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            cam = EigenCAM(
                model=model,
                target_layers=[target_layer],
                reshape_transform=reshape_transform,
            )
            grayscale_cam = cam(input_tensor=pixel_values)[0]
            grayscale_cam_resized = np.array(
                Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize(
                    (image_np.shape[1], image_np.shape[0])
                )
            ) / 255.0

            overlay = show_cam_on_image(image_np, grayscale_cam_resized, use_rgb=True)
            output_path = self.output_dir / f"{image_id}_gradcam_vit.png"
            Image.fromarray(overlay).save(output_path)
            logger.info(f"ViT GradCAM saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ViT GradCAM generation failed: {e}")
        return None
