#!/usr/bin/env python3
"""CLI inference script for the censorship module.

Usage examples:
    python scripts/run_inference.py --image ./test.jpg
    python scripts/run_inference.py --image ./test.jpg --classifier nudenet
    python scripts/run_inference.py --image ./test.jpg --prompt "beach scene"
    python scripts/run_inference.py --prompt "generate explicit content"
    python scripts/run_inference.py --dir ./images/ --output ./results.jsonl
    python scripts/run_inference.py --image ./test.jpg --report --output-dir ./reports
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_inference")


def print_verdict(verdict, verbose: bool = False):
    """Pretty-print a Verdict to stdout."""
    colors = {
        "ALLOW": "\033[92m",   # green
        "BLOCK": "\033[91m",   # red
        "REVIEW": "\033[93m",  # yellow
        "RESET": "\033[0m",
    }
    c = colors.get(verdict.decision, "")
    reset = colors["RESET"]

    max_score = max(verdict.classifier_scores.values()) if verdict.classifier_scores else 0.0
    conf = verdict.reasoner_confidence or max_score

    print(f"\n{'='*60}")
    print(f"Image ID:   {verdict.image_id[:16]}...")
    print(f"Decision:   {c}{verdict.decision}{reset}")
    print(f"Category:   {verdict.primary_category or '—'}")
    print(f"Confidence: {conf:.2f} ({verdict.classifier_model})")
    print(f"Latency:    {verdict.latency_ms:.1f} ms")

    if verdict.reasoner_rationale:
        print(f"Rationale:  {verdict.reasoner_rationale[:200]}...")

    if verbose:
        print("\nCategory scores:")
        for cat, score in sorted(verdict.classifier_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            print(f"  {cat:<20} {score:.3f} {bar}")

        if verdict.prompt_verdict:
            print(f"\nPrompt check: {verdict.prompt_verdict} ({verdict.prompt_category or 'safe'})")

    print(f"{'='*60}\n")


def run_single_image(args, hf_token: str | None = None):
    from censorship.pipeline.combined_pipeline import CombinedPipeline
    from censorship.explainability.report import ReportGenerator
    from censorship.explainability.attention_viz import AttentionVisualizer

    config_path = Path(args.config) if hasattr(args, "config") and args.config else Path("config/models.yaml")
    taxonomy_path = Path("config/taxonomy.yaml")
    policy_path = Path("config/policy_bank.yaml")

    pipeline = CombinedPipeline.from_config(
        config_path=config_path,
        taxonomy_path=taxonomy_path,
        policy_path=policy_path,
        classifier=args.classifier,
        reasoner=args.reasoner if not args.no_reasoner else None,
        prompt_guard=args.prompt_guard if not args.no_prompt_guard else None,
        audit_log=args.audit_log,
        hf_token=hf_token,
    )

    verdict = pipeline.run(
        image_path=args.image if args.image else None,
        prompt=args.prompt if args.prompt else None,
    )
    print_verdict(verdict, verbose=args.verbose)

    if args.report:
        output_dir = Path(args.output_dir) if args.output_dir else Path("reports")
        # Generate score visualization
        viz = AttentionVisualizer(output_dir=output_dir)
        heatmap = viz.generate_simple_heatmap(
            image_path=args.image,
            scores=verdict.classifier_scores,
            image_id=verdict.image_id[:12],
        ) if args.image else None

        report_gen = ReportGenerator(output_dir=output_dir)
        files = report_gen.generate(
            verdict=verdict,
            image_path=args.image,
            heatmap_path=heatmap,
        )
        print(f"Report: {files}")

    return verdict


def run_batch(args, hf_token: str | None = None):
    from censorship.pipeline.image_pipeline import ImagePipeline

    config_path = Path(args.config) if hasattr(args, "config") and args.config else Path("config/models.yaml")

    pipeline = ImagePipeline.from_config(
        config_path=config_path,
        classifier=args.classifier,
        reasoner=args.reasoner if not args.no_reasoner else None,
        audit_log=args.audit_log,
        hf_token=hf_token,
    )

    image_dir = Path(args.dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Processing {len(images)} images...")
    verdicts = pipeline.run_batch(images)

    output_path = Path(args.output) if args.output else Path("results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")

    n_block = sum(1 for v in verdicts if v.decision == "BLOCK")
    n_review = sum(1 for v in verdicts if v.decision == "REVIEW")
    n_allow = sum(1 for v in verdicts if v.decision == "ALLOW")

    print(f"\nResults: ALLOW={n_allow} | REVIEW={n_review} | BLOCK={n_block}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Censorship module — image and prompt safety classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", metavar="PATH", help="Path to single image")
    input_group.add_argument("--dir", metavar="DIR", help="Directory of images (batch mode)")
    parser.add_argument("--prompt", metavar="TEXT", help="Text prompt to check")

    # Model selection
    parser.add_argument("--classifier", default="shieldgemma2",
                        choices=["shieldgemma2", "nudenet", "q16"],
                        help="Layer-1 classifier (default: shieldgemma2)")
    parser.add_argument("--reasoner", default="shieldgemma2_reason",
                        choices=["llavaguard", "shieldgemma2_reason"],
                        help="Layer-2 reasoner (default: shieldgemma2_reason)")
    parser.add_argument("--prompt-guard", default="llamaguard4",
                        dest="prompt_guard",
                        choices=["llamaguard4", "shieldgemma_text"],
                        help="Prompt guard model (default: llamaguard4)")
    parser.add_argument("--no-reasoner", action="store_true",
                        help="Disable Layer-2 reasoner")
    parser.add_argument("--no-prompt-guard", action="store_true",
                        help="Disable prompt guard")

    # Config
    parser.add_argument("--config", default="config/models.yaml",
                        help="Path to models.yaml")
    parser.add_argument("--audit-log", default="audit.jsonl",
                        help="Path to audit log file")

    # Output
    parser.add_argument("--output", metavar="PATH",
                        help="Output path for batch results (jsonl)")
    parser.add_argument("--output-dir", default="reports",
                        help="Directory for reports/heatmaps")
    parser.add_argument("--report", action="store_true",
                        help="Generate JSON+Markdown report")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-category scores")

    args = parser.parse_args()

    if not args.image and not args.dir and not args.prompt:
        parser.error("Provide --image, --dir, or --prompt")

    hf_token = os.environ.get("HF_TOKEN")

    if args.dir:
        run_batch(args, hf_token=hf_token)
    else:
        run_single_image(args, hf_token=hf_token)


if __name__ == "__main__":
    main()
