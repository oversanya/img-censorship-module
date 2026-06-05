"""Оркестратор бенчмарка: загрузка → классификация → метрики → отчёт.

Один публичный вход :func:`run_benchmark`. Грузит курируемый датасет
(`vekshinkir/image-censorship-small`, split `benchmarking`), гоняет каждую
картинку через реальный image-классификатор (`ImageClassifierRunner` =
CLIP zero-shot + NSFW-детектор) с замером латентности, считает метрики и
печатает/сохраняет интерактивный отчёт.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from censor_guard.benchmark import datasets as ds_mod
from censor_guard.benchmark import metrics as metrics_mod
from censor_guard.benchmark import report as report_mod
from censor_guard.evaluation import ImageClassifierRunner
from censor_guard.taxonomy import CATEGORY_SPECS


def _resolve_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _warmup(runner: ImageClassifierRunner) -> float:
    """Прогрев: грузим модели на синтетической картинке. → секунды."""
    t0 = time.time()
    runner.classify(Image.new("RGB", (64, 64), (127, 127, 127)))
    return time.time() - t0


def run_benchmark(
    split: str = ds_mod.DEFAULT_SPLIT,
    limit: int | None = None,
    seed: int = 42,
    save_report: bool = True,
    output_dir: str | Path = "reports",
    make_html: bool = True,
    print_report: bool = True,
) -> dict[str, Any]:
    """Прогнать бенчмарк на курируемом датасете и вернуть пакет метрик.

    Args:
        split: сплит датасета (по умолчанию `benchmarking`).
        limit: ограничить число картинок (None → все 690).
        seed: сид перемешивания при сэмплировании.
        save_report: сохранять ли артефакты (md/html/csv/json).
        output_dir: куда складывать `benchmark_<timestamp>/`.
        make_html: генерировать ли HTML-дашборд (при save_report).
        print_report: печатать ли финальный отчёт в консоль.

    Returns:
        dict с ключами config / latency / overall / per_category / per_dataset /
        by_ai_generated / edge_case / adversarial / report_dir / dataframe / figures.
    """
    token = _resolve_token()
    rows_raw = ds_mod.load_benchmark(split=split, limit=limit, seed=seed, token=token)
    if not rows_raw:
        raise RuntimeError(f"Датасет {ds_mod.DATASET_ID}[{split}] не отдал картинок — проверьте сеть/доступ.")

    runner = ImageClassifierRunner()
    print(f"Прогрев моделей… (block≥{runner.settings.block_threshold}, review≥{runner.settings.review_threshold})")
    warmup_s = _warmup(runner)
    print(f"Модели загружены за {warmup_s:.1f} c.")

    codes = [spec.code for spec in CATEGORY_SPECS]
    rows: list[dict[str, Any]] = []
    bar = tqdm(rows_raw, desc="Классификация", unit="img")
    for src in bar:
        t0 = time.time()
        result = runner.classify(src["image"])
        dt = time.time() - t0
        unsafe = bool(src["label"])
        true_cat = ds_mod.CATEGORY_MAP.get(src["category"]) if unsafe else None
        row: dict[str, Any] = {
            "dataset": src["source"],              # группировка отчёта = источник
            "dataset_category": true_cat or "safe",
            "true_unsafe": unsafe,
            "true_category": true_cat,
            "adversarial": src["source"] == "adversarial",
            "is_edge_case": src["is_edge_case"],
            "ai_generated": src["ai_generated"],
            "source": src["source"],
            "subcategory": src["subcategory"],
            "verdict": result.verdict,
            "flagged": result.flagged,
            "blocked": result.blocked,
            "unsafe_score": result.unsafe_score,
            "pred_categories": list(result.categories),
            "latency_s": dt,
        }
        for code in codes:
            row[f"score_{code}"] = result.score_for(code)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["true_unsafe"] = df["true_unsafe"].astype("object")
    wall_s = float(df["latency_s"].sum())

    config = {
        "dataset_id": ds_mod.DATASET_ID,
        "split": split,
        "limit": limit,
        "seed": seed,
        "block_threshold": runner.settings.block_threshold,
        "review_threshold": runner.settings.review_threshold,
        "visual_model_id": runner.settings.visual_model_id,
        "explicit_model_id": runner.settings.explicit_model_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    metrics = metrics_mod.compute_all(df, warmup_seconds=warmup_s, wall_seconds=wall_s, config=config, skipped=[])

    figs = report_mod.build_figures(
        df, metrics, runner.settings.block_threshold, runner.settings.review_threshold)
    report_dir = None
    if save_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(output_dir) / f"benchmark_{stamp}"
        report_mod.save_report(metrics, df, figs, report_dir, make_html=make_html)

    if print_report:
        print("\n" + report_mod.render_text_report(metrics))
        if report_dir is not None:
            print(f"\nОтчёт сохранён: {report_dir}/  (dashboard.html, report.md, predictions.csv, metrics.json)")

    metrics["report_dir"] = str(report_dir) if report_dir else None
    metrics["dataframe"] = df
    metrics["figures"] = figs
    return metrics
