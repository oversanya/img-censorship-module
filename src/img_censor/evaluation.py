import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List

from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import GuardRequest, Verdict


@dataclass
class EvalRow:
    prompt: str
    input_image: str
    output_image: str
    expected_verdict: str


def load_manifest(path: str) -> List[EvalRow]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            EvalRow(
                prompt=row.get("prompt", ""),
                input_image=row.get("input_image", ""),
                output_image=row.get("output_image", ""),
                expected_verdict=row["expected_verdict"],
            )
            for row in reader
        ]


def evaluate(pipeline: ImageCensorPipeline, rows: Iterable[EvalRow]) -> Dict[str, float]:
    tp = fp = tn = fn = review = 0
    total = 0
    for row in rows:
        total += 1
        result = pipeline.check(
            GuardRequest(
                prompt=row.prompt or None,
                input_image=row.input_image or None,
                output_image=row.output_image or None,
            )
        )
        expected_block = row.expected_verdict == Verdict.BLOCK.value
        predicted_block = result.verdict == Verdict.BLOCK
        if result.verdict == Verdict.REVIEW:
            review += 1

        if expected_block and predicted_block:
            tp += 1
        elif not expected_block and predicted_block:
            fp += 1
        elif not expected_block and not predicted_block:
            tn += 1
        elif expected_block and not predicted_block:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": float(total),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "true_negative": float(tn),
        "false_negative": float(fn),
        "manual_review": float(review),
        "precision_block": precision,
        "recall_block": recall,
        "f1_block": f1,
    }

