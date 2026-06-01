import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from img_censor.config import load_config


GATED_MODELS = {"google/shieldgemma-2-4b-it"}
DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "vocab.*",
    "merges.txt",
    "tokenizer.*",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download enabled Hugging Face models from pipeline config.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Only download detectors that run on this stage. Can be repeated, e.g. --stage prompt.",
    )
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download every file in the model repo instead of only runtime files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_ids = []
    requested_stages = set(args.stage)
    for detector in (config.get("detectors") or {}).values():
        model_id = detector.get("model_id")
        if not model_id:
            continue
        detector_stages = set(detector.get("stages", []))
        if requested_stages and not (requested_stages & detector_stages):
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
        allow_patterns = None if args.all_files else DEFAULT_ALLOW_PATTERNS
        download_with_retries(model_id, cache_dir, args.retries, allow_patterns)
    return 0


def download_with_retries(model_id: str, cache_dir: str, retries: int, allow_patterns) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                max_workers=1,
                allow_patterns=allow_patterns,
            )
            return
        except Exception as error:
            last_error = error
            if attempt == retries:
                break
            delay = min(2**attempt, 10)
            print(f"Download failed for {model_id} on attempt {attempt}/{retries}: {error}")
            print(f"Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(
        f"Could not download {model_id}. Check network access to Hugging Face and retry. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
