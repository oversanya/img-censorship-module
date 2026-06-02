from __future__ import annotations

import argparse
import json

from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the censor module MVP on a local request.")
    parser.add_argument("--scenario", required=True, choices=["text2image", "img2img_stylization", "img2img_editing", "output"])
    parser.add_argument("--stage", default="input", choices=["input", "output"])
    parser.add_argument("--prompt")
    parser.add_argument("--image-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = ModerationRequest(
        scenario=args.scenario,
        stage=args.stage,
        prompt=args.prompt,
        image_path=args.image_path,
    )
    response = GuardrailPipeline().moderate(request)
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

