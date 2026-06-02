"""Attention visualization for VLM-based models (ShieldGemma-2, LlavaGuard).

Extracts cross-attention weights between image patches and model output,
then overlays them on the original image to show which regions drove the decision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Extracts and visualizes attention maps from transformer-based vision models.
    """

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model,
        processor,
        image_path: Union[str, Path],
        prompt: str,
        image_id: str,
        patch_size: int = 14,
    ) -> Path | None:
        """
        Extract attention maps from a VLM and save overlay.

        Args:
            model: HuggingFace transformers VLM
            processor: corresponding processor
            image_path: path to image
            prompt: the prompt used for classification
            image_id: used for output filename
            patch_size: ViT patch size (default 14 for most models)

        Returns:
            Path to saved attention overlay, or None on failure
        """
        try:
            import torch
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            image = Image.open(image_path).convert("RGB")
            W, H = image.size

            inputs = processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Extract last-layer attention: shape (batch, heads, seq, seq)
            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                logger.warning("Model did not return attention weights.")
                return None

            last_attn = outputs.attentions[-1]  # (1, heads, seq, seq)
            # Average over heads
            avg_attn = last_attn[0].mean(dim=0)  # (seq, seq)

            # Find image token positions — they follow the text tokens
            # For multimodal models: first tokens are text, then image patches
            n_patches_h = H // patch_size
            n_patches_w = W // patch_size
            n_image_tokens = n_patches_h * n_patches_w

            total_seq = avg_attn.shape[0]
            n_text_tokens = total_seq - n_image_tokens

            # Attention from last text token to image patches
            if n_text_tokens < 0 or n_image_tokens <= 0:
                logger.warning("Could not identify image token positions.")
                return None

            img_attn = avg_attn[-1, n_text_tokens:n_text_tokens + n_image_tokens]
            img_attn = img_attn.cpu().float().numpy()
            img_attn = img_attn.reshape(n_patches_h, n_patches_w)

            # Normalize and resize to image dimensions
            img_attn = (img_attn - img_attn.min()) / (img_attn.max() - img_attn.min() + 1e-8)
            attn_resized = np.array(
                Image.fromarray((img_attn * 255).astype(np.uint8)).resize(
                    (W, H), resample=Image.BILINEAR
                )
            ) / 255.0

            # Overlay heatmap on original image
            image_np = np.array(image, dtype=np.float32) / 255.0
            heatmap = cm.jet(attn_resized)[:, :, :3]  # drop alpha
            overlay = (0.5 * image_np + 0.5 * heatmap).clip(0, 1)
            overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

            output_path = self.output_dir / f"{image_id}_attention.png"
            overlay_img.save(output_path)
            logger.info(f"Attention map saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Attention visualization failed: {e}")
        return None

    def generate_simple_heatmap(
        self,
        image_path: Union[str, Path],
        scores: dict[str, float],
        image_id: str,
    ) -> Path | None:
        """
        Simple fallback: create a color-coded score bar chart alongside image.
        Used when attention extraction is not possible.
        """
        try:
            import matplotlib.pyplot as plt

            image = Image.open(image_path).convert("RGB")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.imshow(image)
            ax1.set_title("Input Image")
            ax1.axis("off")

            categories = list(scores.keys())
            values = [scores[c] for c in categories]
            colors = ["red" if v >= 0.5 else "orange" if v >= 0.3 else "green" for v in values]
            bars = ax2.barh(categories, values, color=colors)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Confidence Score")
            ax2.set_title("Category Scores")
            ax2.axvline(x=0.5, color="orange", linestyle="--", alpha=0.7, label="Review threshold")
            ax2.axvline(x=0.9, color="red", linestyle="--", alpha=0.7, label="Block threshold")
            ax2.legend(loc="lower right", fontsize=8)

            for bar, val in zip(bars, values):
                ax2.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}", va="center", fontsize=9)

            plt.tight_layout()
            output_path = self.output_dir / f"{image_id}_scores.png"
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            logger.info(f"Score chart saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Score chart generation failed: {e}")
        return None
