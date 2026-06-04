"""Юнит-смоук метрик бенчмарка на синтетической таблице предсказаний.

Не требует сети/моделей: собираем DataFrame руками и проверяем, что
metrics.compute_all и текстовый отчёт считаются и согласованы.
"""

from __future__ import annotations

import math

import pandas as pd

from censor_guard.benchmark import metrics as m
from censor_guard.benchmark.report import render_text_report
from censor_guard.taxonomy import CATEGORY_SPECS


def _row(dataset, dcat, unsafe, tcat, adv, flagged, score, pred, lat):
    row = {
        "dataset": dataset, "dataset_category": dcat,
        "true_unsafe": unsafe, "true_category": tcat, "adversarial": adv,
        "verdict": "block" if flagged else "allow", "flagged": flagged,
        "blocked": flagged, "unsafe_score": score, "pred_categories": pred,
        "latency_s": lat,
    }
    for spec in CATEGORY_SPECS:
        row[f"score_{spec.code}"] = score if spec.code == tcat else 0.0
    return row


def _make_df():
    rows = []
    # 10 sexual unsafe, все пойманы корректной категорией.
    for i in range(10):
        rows.append(_row("nsfw1024", "sexual", True, "sexual", False, True, 0.95, ["sexual"], 0.04))
    # 10 safe, 1 ложно заблокирован.
    for i in range(10):
        flagged = i == 0
        rows.append(_row("tiny_imagenet", "safe", False, None, False, flagged, 0.6 if flagged else 0.2, ["sexual"] if flagged else [], 0.03))
    # 6 violence unsafe, 3 пойманы.
    for i in range(6):
        caught = i < 3
        rows.append(_row("violence_fatima", "violence_gore", True, "violence_gore", False, caught, 0.9 if caught else 0.3, ["violence_gore"] if caught else [], 0.05))
    # 4 adversarial hard-negative (safe), 1 ложно заблокирован.
    for i in range(4):
        flagged = i == 0
        rows.append(_row("meme_sanity", "safe", False, None, True, flagged, 0.7 if flagged else 0.1, [], 0.03))
    df = pd.DataFrame(rows)
    df["true_unsafe"] = df["true_unsafe"].astype("object")
    return df


def test_compute_all_smoke():
    df = _make_df()
    metrics = m.compute_all(df, warmup_seconds=12.0, wall_seconds=1.0,
                            config={"n_dataset": 50, "n_datasets_used": 4,
                                    "block_threshold": 0.85, "review_threshold": 0.55},
                            skipped=[{"dataset": "gore_blood", "reason": "нет HF_TOKEN"}])

    overall = metrics["overall"]
    # 16 unsafe positives, 13 пойманы (10 sexual + 3 violence). negatives=10 safe (adv исключены).
    assert overall["n_positive"] == 16
    assert overall["n_negative"] == 10
    assert overall["tp"] == 13 and overall["fn"] == 3
    assert overall["fp"] == 1 and overall["tn"] == 9
    assert abs(overall["recall"] - 13 / 16) < 1e-6

    sexual = metrics["per_category"]["sexual"]
    assert sexual["supported"] and sexual["recall"] == 1.0
    violence = metrics["per_category"]["violence_gore"]
    assert abs(violence["recall"] - 0.5) < 1e-6

    # Категория без позитивов помечена unsupported.
    assert metrics["per_category"]["self_harm"]["supported"] is False

    # Латентность считается.
    assert metrics["latency"]["n"] == 30
    assert metrics["latency"]["warmup_load_s"] == 12.0

    # Adversarial: 4 hard-negative, FPR = 0.25.
    assert metrics["adversarial"]["n_hard_negative"] == 4
    assert abs(metrics["adversarial"]["hard_negative_fpr"] - 0.25) < 1e-6

    # Отчёт рендерится без исключений и содержит ключевые секции.
    text = render_text_report(metrics)
    assert "ОБЩИЕ МЕТРИКИ" in text and "ПО КАТЕГОРИЯМ" in text and "ИТОГ" in text


def test_verdict_thresholds():
    assert m.category_verdict(0.9).startswith("Хорошо")
    assert m.category_verdict(0.7).startswith("Удовл")
    assert m.category_verdict(0.3).startswith("Слабо")
    assert "FPR" in m.category_verdict(0.9, fpr=0.5)
    assert m.category_verdict(float("nan")) == "н/д"
