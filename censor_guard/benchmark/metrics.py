"""Подсчёт метрик бенчмарка из таблицы предсказаний.

Вход всех функций — `pandas.DataFrame`, который собирает `core.run_benchmark`.
Ожидаемые столбцы:
    dataset, dataset_category, true_unsafe(bool|NA), true_category(str|NA),
    adversarial(bool), flagged(bool), blocked(bool), verdict(str),
    unsafe_score(float), latency_s(float),
    pred_categories(list[str]), score_<code>(float) для каждого кода таксономии.

Все функции возвращают обычные JSON-сериализуемые dict/списки, чтобы их можно
было и напечатать, и сохранить в `metrics.json`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from censor_guard.taxonomy import CATEGORY_BY_CODE, CATEGORY_SPECS


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _round(x: float, n: int = 4) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return round(float(x), n)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC с защитой от вырожденных случаев (один класс)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score

    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def binary_block(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Полный блок бинарных метрик (Unsafe = положительный класс)."""
    from sklearn.metrics import confusion_matrix

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    n = len(y_true)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "n": int(n),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "accuracy": _round(_safe_div(tp + tn, n)),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(_safe_div(2 * precision * recall, precision + recall)),
        "specificity": _round(_safe_div(tn, tn + fp)),
        "fpr": _round(_safe_div(fp, fp + tn)),
        "roc_auc": _round(_roc_auc(y_true, scores)),
        "pr_auc": _round(_pr_auc(y_true, scores)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def overall_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Общие метрики Safe vs Unsafe по всем размеченным НЕ-adversarial строкам."""
    mask = df["true_unsafe"].notna() & (~df["adversarial"].astype(bool))
    sub = df[mask]
    if sub.empty:
        return {"n": 0, "note": "нет размеченных строк"}
    return binary_block(
        sub["true_unsafe"].to_numpy(dtype=bool),
        sub["flagged"].to_numpy(dtype=bool),
        sub["unsafe_score"].to_numpy(dtype=float),
    )


def per_category_metrics(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Метрики по каждому коду таксономии (one-vs-rest).

    Положительные = unsafe-картинки с `true_category == code`.
    Отрицательные = все safe-негативы (true_unsafe == False, не adversarial).
    Скор = `score_<code>` (слитая оценка движка по этой категории).
    Доп. «attribution» — доля пойманных позитивов, где код реально попал в
    предсказанные категории движка.
    """
    neg_mask = (df["true_unsafe"] == False) & (~df["adversarial"].astype(bool))  # noqa: E712
    negatives = df[neg_mask]

    out: dict[str, dict[str, Any]] = {}
    for spec in CATEGORY_SPECS:
        code = spec.code
        score_col = f"score_{code}"
        if score_col not in df.columns:
            continue
        pos = df[(df["true_unsafe"] == True) & (df["true_category"] == code)]  # noqa: E712
        if pos.empty:
            out[code] = {"label": spec.label, "n_positive": 0, "supported": False}
            continue

        pool = pd.concat([pos, negatives], ignore_index=True)
        y_true = (pool["true_unsafe"] == True).to_numpy(dtype=bool)  # noqa: E712
        scores = pool[score_col].to_numpy(dtype=float)
        # Предсказание категории = код попал в pred_categories (block/review движка).
        y_pred = pool["pred_categories"].apply(lambda cs: code in (cs or [])).to_numpy(dtype=bool)

        block = binary_block(y_true, y_pred, scores)
        caught = pos["pred_categories"].apply(lambda cs: code in (cs or [])).mean()
        flagged_recall = pos["flagged"].mean()  # поймано как unsafe (любой категорией)
        out[code] = {
            "label": spec.label,
            "hard_block": spec.hard_block,
            "supported": True,
            "n_positive": int(len(pos)),
            "n_negative": int(len(negatives)),
            "recall": block["recall"],            # детект именно этой категорией
            "flagged_recall": _round(flagged_recall),  # поймано как unsafe вообще
            "precision": block["precision"],
            "f1": block["f1"],
            "fpr": block["fpr"],
            "roc_auc": block["roc_auc"],
            "attribution": _round(caught),        # = recall, дублируем по смыслу
        }
    return out


def per_dataset_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Сводка по каждому датасету."""
    rows: list[dict[str, Any]] = []
    for name, sub in df.groupby("dataset", sort=False):
        labeled = sub[sub["true_unsafe"].notna()]
        has_both = labeled["true_unsafe"].nunique() > 1 if not labeled.empty else False
        row: dict[str, Any] = {
            "dataset": name,
            "category": sub["dataset_category"].iloc[0],
            "n": int(len(sub)),
            "mean_unsafe_score": _round(sub["unsafe_score"].mean()),
            "mean_latency_ms": _round(sub["latency_s"].mean() * 1000, 1),
            "adversarial": bool(sub["adversarial"].any()),
        }
        if has_both:
            block = binary_block(
                labeled["true_unsafe"].to_numpy(dtype=bool),
                labeled["flagged"].to_numpy(dtype=bool),
                labeled["unsafe_score"].to_numpy(dtype=float),
            )
            row.update(accuracy=block["accuracy"], recall=block["recall"],
                       precision=block["precision"], f1=block["f1"], fpr=block["fpr"],
                       roc_auc=block["roc_auc"])
        elif not labeled.empty and bool(labeled["true_unsafe"].iloc[0]):
            row["recall"] = _round(labeled["flagged"].mean())  # all-unsafe → recall
        elif not labeled.empty:
            # all-safe → доля ошибочных срабатываний (FPR).
            row["fpr"] = _round(labeled["flagged"].mean())
        rows.append(row)
    return rows


def adversarial_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Отдельный учёт adversarial-строк (hard-negatives / обфускация)."""
    adv = df[df["adversarial"].astype(bool)]
    if adv.empty:
        return {"n": 0}
    out: dict[str, Any] = {"n": int(len(adv))}
    neg = adv[adv["true_unsafe"] == False]  # noqa: E712
    pos = adv[adv["true_unsafe"] == True]  # noqa: E712
    if not neg.empty:
        out["hard_negative_fpr"] = _round(neg["flagged"].mean())  # ложные блокировки
        out["n_hard_negative"] = int(len(neg))
    if not pos.empty:
        out["obfuscated_recall"] = _round(pos["flagged"].mean())  # удержано unsafe
        out["n_obfuscated"] = int(len(pos))
    return out


def latency_metrics(df: pd.DataFrame, warmup_seconds: float, wall_seconds: float) -> dict[str, Any]:
    lat = df["latency_s"].to_numpy(dtype=float)
    lat = lat[lat > 0]
    if lat.size == 0:
        return {"n": 0}
    return {
        "n": int(lat.size),
        "warmup_load_s": _round(warmup_seconds, 2),
        "wall_seconds": _round(wall_seconds, 1),
        "mean_ms": _round(lat.mean() * 1000, 1),
        "median_ms": _round(np.median(lat) * 1000, 1),
        "p90_ms": _round(np.percentile(lat, 90) * 1000, 1),
        "p95_ms": _round(np.percentile(lat, 95) * 1000, 1),
        "p99_ms": _round(np.percentile(lat, 99) * 1000, 1),
        "throughput_img_s": _round(_safe_div(lat.size, lat.sum()), 1),
    }


def compute_all(df: pd.DataFrame, *, warmup_seconds: float, wall_seconds: float,
                config: dict[str, Any], skipped: list[dict[str, str]]) -> dict[str, Any]:
    """Собрать полный пакет метрик для отчёта/сохранения."""
    return {
        "config": config,
        "n_images": int(len(df)),
        "skipped_datasets": skipped,
        "latency": latency_metrics(df, warmup_seconds, wall_seconds),
        "overall": overall_metrics(df),
        "per_category": per_category_metrics(df),
        "per_dataset": per_dataset_metrics(df),
        "adversarial": adversarial_metrics(df),
    }


def category_verdict(recall: float, fpr: float | None = None) -> str:
    """Вербальная оценка работоспособности по категории."""
    if recall is None or (isinstance(recall, float) and math.isnan(recall)):
        return "н/д"
    if recall >= 0.85:
        base = "Хорошо ✅"
    elif recall >= 0.60:
        base = "Удовл. ⚠️"
    else:
        base = "Слабо ❌"
    if fpr is not None and not math.isnan(fpr) and fpr > 0.30:
        base += " (высокий FPR)"
    return base
