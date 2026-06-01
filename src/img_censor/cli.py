import argparse
import json
import sys
from typing import Optional

from img_censor.config import load_config
from img_censor.mock import mock_check
from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import GuardRequest


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the image censorship pipeline.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline YAML config.")
    parser.add_argument("--prompt", default=None, help="User text prompt.")
    parser.add_argument("--input-image", default=None, help="Input image path or URL for img2img.")
    parser.add_argument("--output-image", default=None, help="Generated output image path or URL.")
    parser.add_argument("--request-id", default=None, help="Optional external request id for audit logs.")
    parser.add_argument("--interactive", action="store_true", help="Read prompts from terminal until exit.")
    parser.add_argument("--mock", action="store_true", help="Run without loading Hugging Face models.")
    parser.add_argument("--describe", action="store_true", help="Print configured detectors and exit.")
    parser.add_argument("--block-threshold", type=float, default=None, help="Override global block threshold.")
    parser.add_argument("--review-threshold", type=float, default=None, help="Override global review threshold.")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.block_threshold is not None:
        config.setdefault("decision", {})["block_threshold"] = args.block_threshold
        for detector_config in (config.get("detectors") or {}).values():
            detector_config["block_threshold"] = args.block_threshold
    if args.review_threshold is not None:
        config.setdefault("decision", {})["review_threshold"] = args.review_threshold
        for detector_config in (config.get("detectors") or {}).values():
            detector_config["review_threshold"] = args.review_threshold

    if args.describe:
        pipeline = ImageCensorPipeline(config)
        print(json.dumps(pipeline.describe(), ensure_ascii=False, indent=2))
        return 0

    checker = build_checker(config, args.mock)

    if args.interactive:
        return run_interactive(checker, args)

    prompt = args.prompt
    if prompt is None and not args.input_image and not args.output_image:
        prompt = read_prompt_once()

    request = GuardRequest(
        prompt=prompt,
        input_image=args.input_image,
        output_image=args.output_image,
        request_id=args.request_id,
    )

    result = checker(request)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def build_checker(config: dict, use_mock: bool):
    if use_mock:
        return mock_check
    pipeline = ImageCensorPipeline(config)
    return pipeline.check


def read_prompt_once() -> str:
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return input("Введите prompt для проверки: ").strip()


def run_interactive(checker, args: argparse.Namespace) -> int:
    print("Интерактивный режим цензор-модуля. Введите prompt, пустая строка или Ctrl-D для выхода.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            return 0

        request = GuardRequest(
            prompt=prompt,
            input_image=args.input_image,
            output_image=args.output_image,
            request_id=args.request_id,
        )
        result = checker(request)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
