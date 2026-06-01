import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from img_censor.config import load_config


GATED_MODELS = {"google/shieldgemma-2-4b-it"}


def parse_args():
    parser = argparse.ArgumentParser(description="Download enabled Hugging Face models from pipeline config.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_ids = []
    for detector in (config.get("detectors") or {}).values():
        model_id = detector.get("model_id")
        if not model_id:
            continue
        if not args.include_disabled and not detector.get("enabled", False):
            continue
        if model_id in GATED_MODELS and not args.include_gated:
            continue
        model_ids.append(model_id)

    cache_dir = args.cache_dir or (config.get("runtime") or {}).get("cache_dir") or "models/hf-cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for model_id in sorted(set(model_ids)):
        print(f"Downloading {model_id} -> cache {cache_dir}")
        snapshot_download(repo_id=model_id, cache_dir=cache_dir, max_workers=4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
