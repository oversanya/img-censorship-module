#!/usr/bin/env python3
"""Pre-download all required models to HuggingFace cache.

Run this once before inference to avoid slow first-run downloads:
    python scripts/download_models.py
    python scripts/download_models.py --models shieldgemma2 nudenet
    python scripts/download_models.py --skip-gated  # skip models requiring HF token
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("download_models")


MODEL_SPECS = {
    "shieldgemma2": {
        "hf_id": "google/shieldgemma-2-4b-it",
        "type": "hf",
        "gated": False,
        "size_gb": 8.5,
        "description": "ShieldGemma-2 4B — primary image classifier + fallback reasoner",
    },
    "nudenet": {
        "hf_id": None,
        "type": "pip",
        "gated": False,
        "size_gb": 0.1,
        "description": "NudeNet — fast NSFW detector (downloads on first use via pip)",
    },
    "q16": {
        "hf_id": "Falconsai/nsfw_image_detection",
        "type": "hf",
        "gated": False,
        "size_gb": 0.33,
        "description": "Falconsai ViT NSFW — fast CPU classifier (~330 MB)",
    },
    "llavaguard": {
        "hf_id": "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV",
        "type": "hf",
        "gated": False,
        "size_gb": 1.0,
        "description": "LlavaGuard-0.5B — compact VLM reasoner with rationale (~1 GB)",
    },
    "llamaguard4": {
        "hf_id": "meta-llama/Llama-Guard-3-1B",
        "type": "hf",
        "gated": True,
        "size_gb": 2.0,
        "description": "Llama-Guard-3-1B — text prompt guard, requires HF token",
    },
}


def download_hf_model(hf_id: str, hf_token: str | None, dry_run: bool = False) -> bool:
    """Download a HuggingFace model to cache."""
    logger.info(f"Downloading {hf_id} ...")
    if dry_run:
        logger.info(f"[DRY RUN] Would download: {hf_id}")
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=hf_id,
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        logger.info(f"✓ {hf_id} downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {hf_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download censorship module models")
    parser.add_argument(
        "--models", nargs="*", choices=list(MODEL_SPECS), default=list(MODEL_SPECS),
        help="Models to download (default: all)"
    )
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip models requiring HF token / access approval")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Gated models will fail to download.")

    total_size = 0.0
    to_download = []

    for name in args.models:
        spec = MODEL_SPECS[name]
        if args.skip_gated and spec.get("gated"):
            logger.info(f"Skipping gated model: {name}")
            continue
        if spec["type"] == "pip":
            logger.info(f"Skipping {name} (downloads on first use via pip)")
            continue
        to_download.append(name)
        total_size += spec.get("size_gb", 0)

    print(f"\nModels to download: {to_download}")
    print(f"Estimated disk space: {total_size:.1f} GB\n")

    if not to_download:
        print("Nothing to download.")
        return

    results = {}
    for name in to_download:
        spec = MODEL_SPECS[name]
        print(f"\n[{name}] {spec['description']}")
        print(f"  HF ID: {spec['hf_id']} | Size: ~{spec['size_gb']} GB")
        ok = download_hf_model(spec["hf_id"], hf_token, dry_run=args.dry_run)
        results[name] = ok

    print("\n--- Summary ---")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")


if __name__ == "__main__":
    main()
