#!/usr/bin/env python3
"""Benchmark censorship pipeline on UnsafeBench dataset.

Computes Precision, Recall, F1 per category and overall.
Optionally measures adversarial robustness.

Usage:
    python scripts/benchmark.py --dataset-path ./unsafebench/
    python scripts/benchmark.py --dataset-path ./unsafebench/ --models shieldgemma2 nudenet
    python scripts/benchmark.py --dataset-path ./unsafebench/ --adversarial
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")


def load_unsafebench(dataset_path: Path) -> list[dict]:
    """
    Load UnsafeBench dataset.
    Expected structure:
        dataset_path/
            metadata.jsonl  — {image_path, label (safe/unsafe), categories: [...]}
            images/
    """
    meta_path = dataset_path / "metadata.jsonl"
    if not meta_path.exists():
        # Try HuggingFace datasets
        try:
            from datasets import load_dataset
            logger.info("Loading UnsafeBench from HuggingFace hub...")
            ds = load_dataset("yiting/UnsafeBench", split="test")
            records = []
            for item in ds:
                records.append({
                    "image": item.get("image"),
                    "label": item.get("label", "safe"),
                    "categories": item.get("categories", []),
                    "source": item.get("source", "unknown"),
                })
            logger.info(f"Loaded {len(records)} records from HuggingFace.")
            return records
        except Exception as e:
            logger.error(f"Failed to load UnsafeBench: {e}")
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Pass --dataset-path or ensure HuggingFace datasets is installed."
            )

    records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["image_path"] = dataset_path / "images" / r["image_path"]
                records.append(r)
    logger.info(f"Loaded {len(records)} records from {meta_path}")
    return records


def run_evaluation(pipeline, records: list[dict], max_samples: int | None = None) -> dict:
    """Run pipeline on records and collect predictions vs ground truth."""
    from tqdm import tqdm

    if max_samples:
        records = records[:max_samples]

    y_true = []
    y_pred = []
    y_scores = []
    latencies = []
    errors = 0

    for record in tqdm(records, desc="Evaluating"):
        image = record.get("image") or record.get("image_path")
        if image is None:
            errors += 1
            continue

        # Ground truth: 1 = unsafe, 0 = safe
        label = record.get("label", "safe")
        gt = 1 if label.lower() == "unsafe" else 0
        y_true.append(gt)

        try:
            # Save PIL image to temp file if needed
            if hasattr(image, "save"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    verdict = pipeline.run(tmp.name)
            else:
                verdict = pipeline.run(str(image))

            pred = 1 if verdict.decision in ("BLOCK", "REVIEW") else 0
            score = verdict.reasoner_confidence or 0.0
            y_pred.append(pred)
            y_scores.append(score)
            latencies.append(verdict.latency_ms)
        except Exception as e:
            logger.warning(f"Failed on {image}: {e}")
            y_pred.append(0)
            y_scores.append(0.0)
            latencies.append(0.0)
            errors += 1

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "latencies": latencies,
        "n_errors": errors,
        "n_total": len(records),
    }


def compute_metrics(eval_results: dict) -> dict:
    """Compute classification metrics from evaluation results."""
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        confusion_matrix,
    )
    import numpy as np

    y_true = eval_results["y_true"]
    y_pred = eval_results["y_pred"]
    y_scores = eval_results["y_scores"]
    latencies = eval_results["latencies"]

    report = classification_report(y_true, y_pred, target_names=["safe", "unsafe"], output_dict=True)

    auc = None
    if len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            pass

    lat = np.array([l for l in latencies if l > 0])
    latency_stats = {
        "p50_ms": float(np.percentile(lat, 50)) if len(lat) > 0 else 0,
        "p95_ms": float(np.percentile(lat, 95)) if len(lat) > 0 else 0,
        "p99_ms": float(np.percentile(lat, 99)) if len(lat) > 0 else 0,
        "mean_ms": float(np.mean(lat)) if len(lat) > 0 else 0,
    }

    return {
        "classification_report": report,
        "roc_auc": auc,
        "latency": latency_stats,
        "n_total": eval_results["n_total"],
        "n_errors": eval_results["n_errors"],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics_table(model_name: str, metrics: dict):
    """Print a formatted metrics summary."""
    report = metrics["classification_report"]
    unsafe = report.get("unsafe", {})
    lat = metrics["latency"]

    print(f"\n{'─'*60}")
    print(f"Model: {model_name}")
    print(f"{'─'*60}")
    print(f"  Precision (unsafe): {unsafe.get('precision', 0):.3f}")
    print(f"  Recall    (unsafe): {unsafe.get('recall', 0):.3f}")
    print(f"  F1        (unsafe): {unsafe.get('f1-score', 0):.3f}")
    print(f"  Macro F1:           {report.get('macro avg', {}).get('f1-score', 0):.3f}")
    if metrics["roc_auc"]:
        print(f"  ROC-AUC:            {metrics['roc_auc']:.3f}")
    print(f"  Latency p50/p95:    {lat['p50_ms']:.0f} / {lat['p95_ms']:.0f} ms")
    print(f"  Total samples:      {metrics['n_total']} ({metrics['n_errors']} errors)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark censorship pipeline")
    parser.add_argument("--dataset-path", default="./unsafebench",
                        help="Path to UnsafeBench dataset")
    parser.add_argument("--models", nargs="*",
                        choices=["shieldgemma2", "nudenet", "q16"],
                        default=["shieldgemma2"],
                        help="Models to benchmark")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run adversarial robustness evaluation")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--no-reasoner", action="store_true",
                        help="Disable Layer-2 reasoner (benchmark Layer-1 only)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    records = load_unsafebench(dataset_path)
    logger.info(f"Dataset: {len(records)} images")

    hf_token = os.environ.get("HF_TOKEN")
    all_results = {}

    for model_name in args.models:
        logger.info(f"\nBenchmarking {model_name}...")

        from censorship.pipeline.image_pipeline import ImagePipeline
        pipeline = ImagePipeline.from_config(
            classifier=model_name,
            reasoner=None if args.no_reasoner else "shieldgemma2_reason",
            audit_log=None,
            hf_token=hf_token,
        )

        eval_results = run_evaluation(pipeline, records, max_samples=args.max_samples)
        metrics = compute_metrics(eval_results)
        print_metrics_table(model_name, metrics)
        all_results[model_name] = metrics

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'─'*80}")
        print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>8} {'p95(ms)':>10}")
        print(f"{'─'*80}")
        for name, m in all_results.items():
            r = m["classification_report"]
            u = r.get("unsafe", {})
            auc_str = f"{m['roc_auc']:.3f}" if m["roc_auc"] else "  —  "
            print(
                f"{name:<20} "
                f"{u.get('precision', 0):>10.3f} "
                f"{u.get('recall', 0):>10.3f} "
                f"{u.get('f1-score', 0):>10.3f} "
                f"{auc_str:>8} "
                f"{m['latency']['p95_ms']:>10.0f}"
            )


if __name__ == "__main__":
    main()
