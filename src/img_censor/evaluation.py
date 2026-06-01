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
    expected_category: str = ""
    attack_type: str = "benign"


def load_manifest(path: str) -> List[EvalRow]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            EvalRow(
                prompt=row.get("prompt", ""),
                input_image=row.get("input_image", ""),
                output_image=row.get("output_image", ""),
                expected_verdict=row["expected_verdict"],
                expected_category=row.get("expected_category", ""),
                attack_type=row.get("attack_type", "benign"),
            )
            for row in reader
        ]


def evaluate(pipeline: ImageCensorPipeline, rows: Iterable[EvalRow]) -> Dict[str, object]:
    tp = fp = tn = fn = review = 0
    total = 0
    category_expected = {}
    category_hit = {}
    attack_expected = {}
    attack_hit = {}
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
        expected_category = row.expected_category
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

        if expected_block and expected_category:
            category_expected[expected_category] = category_expected.get(expected_category, 0) + 1
            if expected_category in result.categories:
                category_hit[expected_category] = category_hit.get(expected_category, 0) + 1

        if expected_block and row.attack_type:
            attack_expected[row.attack_type] = attack_expected.get(row.attack_type, 0) + 1
            if predicted_block:
                attack_hit[row.attack_type] = attack_hit.get(row.attack_type, 0) + 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    manual_review_rate = review / total if total else 0.0
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
        "false_positive_rate": false_positive_rate,
        "manual_review_rate": manual_review_rate,
        "category_recall": {
            category: category_hit.get(category, 0) / count
            for category, count in sorted(category_expected.items())
        },
        "attack_recall": {
            attack_type: attack_hit.get(attack_type, 0) / count
            for attack_type, count in sorted(attack_expected.items())
        },
    }
