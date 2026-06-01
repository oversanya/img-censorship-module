import argparse
import json

from img_censor.config import load_config
from img_censor.evaluation import evaluate, load_manifest
from img_censor.pipeline import ImageCensorPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate censor pipeline on a CSV manifest.")
    parser.add_argument("manifest", help="CSV with prompt,input_image,output_image,expected_verdict columns.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = ImageCensorPipeline(load_config(args.config))
    metrics = evaluate(pipeline, load_manifest(args.manifest))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

