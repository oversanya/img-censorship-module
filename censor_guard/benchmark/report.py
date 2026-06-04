"""Графики, консольный отчёт, Markdown и HTML-дашборд по результатам бенчмарка.

Графики строятся через matplotlib в headless-режиме (Agg) и сохраняются в PNG;
те же фигуры возвращаются как объекты, чтобы ноутбук мог показать их inline.
Консольный отчёт печатается всегда; Markdown/HTML/CSV — только при save_report.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.figure as fig

matplotlib.use("Agg")  # без дисплея — пишем в файлы/буфер
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from censor_guard.benchmark.metrics import category_verdict  # noqa: E402

plt.rcParams["figure.dpi"] = 110


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "н/д"
        return f"{x:.3f}"
    return str(x)


def _ascii_table(headers: list[str], rows: list[list[Any]]) -> str:
    cols = [headers] + [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(str(cols[r][c])) for r in range(len(cols))) for c in range(len(headers))]
    line = "─┼─".join("─" * w for w in widths)
    out = [" │ ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)), line]
    for r in rows:
        out.append(" │ ".join(_fmt(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Графики.
# ─────────────────────────────────────────────────────────────────────────────
def build_figures(df: pd.DataFrame, metrics: dict[str, Any], block_thr: float, review_thr: float) -> dict[str, "fig.Figure"]:
    figs: dict[str, fig.Figure] = {}
    labeled = df[df["true_unsafe"].notna() & (~df["adversarial"].astype(bool))]

    # (1) Confusion matrix + (2) распределение unsafe_score.
    if not labeled.empty:
        from sklearn.metrics import confusion_matrix

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        cm = confusion_matrix(labeled["true_unsafe"].astype(bool), labeled["flagged"].astype(bool), labels=[False, True])
        im = ax[0].imshow(cm, cmap="Blues")
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["pred Safe", "pred Unsafe"])
        ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["true Safe", "true Unsafe"])
        for (r, c), v in np.ndenumerate(cm):
            ax[0].text(c, r, str(v), ha="center", va="center",
                       color="white" if v > cm.max() / 2 else "black", fontsize=13)
        ax[0].set_title("Confusion matrix (Safe vs Unsafe)")
        fig.colorbar(im, ax=ax[0], fraction=0.046)

        for val, color, name in [(False, "#2e7d32", "Safe"), (True, "#c62828", "Unsafe")]:
            sub = labeled[labeled["true_unsafe"] == val]["unsafe_score"]
            ax[1].hist(sub, bins=20, alpha=0.6, color=color, label=name, range=(0, 1))
        ax[1].axvline(block_thr, ls="--", color="black", alpha=0.7, label=f"block {block_thr}")
        ax[1].axvline(review_thr, ls=":", color="grey", label=f"review {review_thr}")
        ax[1].set_xlabel("unsafe_score"); ax[1].set_ylabel("кол-во"); ax[1].legend()
        ax[1].set_title("Распределение unsafe_score")
        fig.tight_layout()
        figs["overview"] = fig

    # (3) ROC overall + overlay по категориям.
    if not labeled.empty and labeled["true_unsafe"].nunique() > 1:
        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(figsize=(6, 5.2))
        fpr, tpr, _ = roc_curve(labeled["true_unsafe"].astype(bool), labeled["unsafe_score"])
        auc = metrics["overall"].get("roc_auc")
        ax.plot(fpr, tpr, color="#1565c0", lw=2.2, label=f"overall (AUC={_fmt(auc)})")
        negatives = df[(df["true_unsafe"] == False) & (~df["adversarial"].astype(bool))]  # noqa: E712
        for code, cm in metrics["per_category"].items():
            if not cm.get("supported"):
                continue
            pos = df[(df["true_unsafe"] == True) & (df["true_category"] == code)]  # noqa: E712
            score_col = f"score_{code}"
            pool = pd.concat([pos, negatives], ignore_index=True)
            if pool["true_unsafe"].nunique() < 2:
                continue
            fpr_c, tpr_c, _ = roc_curve((pool["true_unsafe"] == True).astype(bool), pool[score_col])  # noqa: E712
            ax.plot(fpr_c, tpr_c, lw=1.1, alpha=0.7, label=f"{code} ({_fmt(cm.get('roc_auc'))})")
        ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC: общий и по категориям")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        figs["roc"] = fig

    # (4) Per-category recall/precision/f1.
    cats = [(c, m) for c, m in metrics["per_category"].items() if m.get("supported")]
    if cats:
        fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.1), 4.6))
        labels = [c for c, _ in cats]
        x = np.arange(len(labels))
        w = 0.26
        for off, key, color in [(-w, "recall", "#c62828"), (0, "precision", "#1565c0"), (w, "f1", "#2e7d32")]:
            vals = [(_nan0(m.get(key))) for _, m in cats]
            ax.bar(x + off, vals, w, label=key, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.legend()
        ax.set_title("Метрики по категориям таксономии")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        figs["per_category"] = fig

    # (5) Per-dataset recall.
    pdm = [r for r in metrics["per_dataset"] if "recall" in r]
    if pdm:
        fig, ax = plt.subplots(figsize=(max(7, len(pdm) * 0.9), 4.4))
        names = [r["dataset"] for r in pdm]
        vals = [_nan0(r["recall"]) for r in pdm]
        xpos = np.arange(len(names))
        ax.bar(xpos, vals, color="#6a1b9a", alpha=0.85)
        ax.set_ylim(0, 1); ax.set_ylabel("recall")
        ax.set_xticks(xpos)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title("Recall по датасетам (где определён)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        figs["per_dataset"] = fig

    # (6) Латентность.
    lat = df["latency_s"].to_numpy(dtype=float) * 1000
    lat = lat[lat > 0]
    if lat.size:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].hist(lat, bins=30, color="#00838f", alpha=0.85)
        ax[0].set_xlabel("latency, мс"); ax[0].set_ylabel("кол-во")
        ax[0].set_title("Распределение latency")
        ax[1].boxplot(lat, vert=False, showfliers=False)
        ax[1].set_xlabel("latency, мс"); ax[1].set_yticks([])
        ax[1].set_title("Boxplot latency")
        fig.tight_layout()
        figs["latency"] = fig

    return figs


def _nan0(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(x)


def show_figures(figs: dict[str, "fig.Figure"]) -> None:
    """Показать фигуры inline в Jupyter — backend-независимо (через PNG-байты).

    Нужно потому, что бенчмарк форсит headless-backend Agg (для CLI), а под Agg
    у Figure нет inline-репрезентации, и обычный `display(fig)` покажет текст
    `<Figure …>`, а не картинку. Здесь рендерим фигуру в PNG и отдаём как Image.
    """
    try:
        from IPython.display import Image as IPImage, display
    except Exception:
        return
    for name, fig in figs.items():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        print(f"=== {name} ===")
        display(IPImage(data=buf.getvalue()))


def save_figures(figs: dict[str, "fig.Figure"], figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, fig in figs.items():
        p = figures_dir / f"{name}.png"
        fig.savefig(p, bbox_inches="tight")
        paths[name] = p
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Текстовый отчёт (консоль + Markdown).
# ─────────────────────────────────────────────────────────────────────────────
def render_text_report(metrics: dict[str, Any]) -> str:
    cfg = metrics["config"]
    lat = metrics["latency"]
    overall = metrics["overall"]
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("ОТЧЁТ БЕНЧМАРКА ЦЕНЗОР-МОДУЛЯ")
    add("=" * 78)
    add(f"Картинок на датасет : {cfg.get('n_dataset')}")
    add(f"Всего изображений   : {metrics['n_images']}")
    add(f"Датасетов использовано: {cfg.get('n_datasets_used')} | пропущено: {len(metrics['skipped_datasets'])}")
    add(f"Пороги движка       : block ≥ {cfg.get('block_threshold')} | review ≥ {cfg.get('review_threshold')}")
    add("")

    # Латентность.
    add("─ ЛАТЕНТНОСТЬ " + "─" * 64)
    if lat.get("n"):
        add(f"Загрузка моделей (warmup): {lat['warmup_load_s']} c (разовая)")
        add(f"Полное время прогона     : {lat['wall_seconds']} c на {lat['n']} картинок")
        add(f"На изображение           : mean {lat['mean_ms']} мс | median {lat['median_ms']} мс | "
            f"p95 {lat['p95_ms']} мс | p99 {lat['p99_ms']} мс")
        add(f"Пропускная способность   : {lat['throughput_img_s']} img/s")
    add("")

    # Общие метрики.
    add("─ ОБЩИЕ МЕТРИКИ (Safe vs Unsafe) " + "─" * 45)
    if overall.get("n"):
        add(_ascii_table(
            ["accuracy", "precision", "recall", "f1", "FPR", "ROC-AUC", "PR-AUC"],
            [[overall["accuracy"], overall["precision"], overall["recall"], overall["f1"],
              overall["fpr"], overall["roc_auc"], overall["pr_auc"]]],
        ))
        add(f"Матрица: TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} TN={overall['tn']} "
            f"(positives={overall['n_positive']}, negatives={overall['n_negative']})")
    else:
        add("нет размеченных строк")
    add("")

    # По категориям.
    add("─ ПО КАТЕГОРИЯМ ТАКСОНОМИИ " + "─" * 51)
    rows = []
    for code, m in metrics["per_category"].items():
        if not m.get("supported"):
            rows.append([code, m.get("n_positive", 0), "—", "—", "—", "—", "нет данных"])
            continue
        rows.append([
            code, m["n_positive"], m["recall"], m["precision"], m["fpr"], m["roc_auc"],
            category_verdict(m["recall"], m.get("fpr")),
        ])
    add(_ascii_table(["категория", "n+", "recall", "precision", "FPR", "ROC-AUC", "оценка"], rows))
    add("")

    # По датасетам.
    add("─ ПО ДАТАСЕТАМ " + "─" * 63)
    rows = []
    for r in metrics["per_dataset"]:
        rows.append([
            r["dataset"], r["category"], r["n"],
            r.get("recall", r.get("fpr", "—")),
            r.get("precision", "—"), r.get("roc_auc", "—"),
            r["mean_latency_ms"],
        ])
    add(_ascii_table(["датасет", "класс", "n", "recall/fpr", "precision", "ROC-AUC", "lat,мс"], rows))
    add("")

    # Adversarial.
    adv = metrics["adversarial"]
    if adv.get("n"):
        add("─ ADVERSARIAL / HARD-NEGATIVE " + "─" * 48)
        if "hard_negative_fpr" in adv:
            add(f"Counterfactual hard-negatives (ожидаем allow): FPR = {adv['hard_negative_fpr']} "
                f"на {adv['n_hard_negative']} картинках")
        if "obfuscated_recall" in adv:
            add(f"Обфусцированный unsafe (ожидаем block): recall = {adv['obfuscated_recall']} "
                f"на {adv['n_obfuscated']} картинках")
        add("")

    # Пропущенные.
    if metrics["skipped_datasets"]:
        add("─ ПРОПУЩЕННЫЕ ДАТАСЕТЫ " + "─" * 55)
        for s in metrics["skipped_datasets"]:
            add(f"  • {s['dataset']}: {s['reason']}")
        add("")

    # Итоговая оценка.
    add("─ ИТОГ " + "─" * 71)
    add(_overall_assessment(metrics))
    add("=" * 78)
    return "\n".join(lines)


def _overall_assessment(metrics: dict[str, Any]) -> str:
    overall = metrics["overall"]
    if not overall.get("n"):
        return "Недостаточно размеченных данных для общей оценки."
    recall = overall["recall"]
    fpr = overall["fpr"]
    auc = overall["roc_auc"]
    parts = [
        f"Общий разделяющий потенциал ROC-AUC={_fmt(auc)}, recall={_fmt(recall)}, FPR={_fmt(fpr)}.",
    ]
    if recall >= 0.85 and fpr <= 0.15:
        parts.append("Классификатор уверенно ловит небезопасный контент при низком уровне ложных блокировок — готов к пилоту.")
    elif recall >= 0.7:
        parts.append("Детект приемлемый, но стоит проверить категории с низким recall и/или повышенным FPR ниже.")
    else:
        parts.append("Детект слабый — рекомендуется калибровка порогов и усиление слабых категорий перед использованием.")

    weak = [c for c, m in metrics["per_category"].items()
            if m.get("supported") and not _is_nan(m["recall"]) and m["recall"] < 0.6]
    strong = [c for c, m in metrics["per_category"].items()
              if m.get("supported") and not _is_nan(m["recall"]) and m["recall"] >= 0.85]
    if strong:
        parts.append("Сильные категории: " + ", ".join(strong) + ".")
    if weak:
        parts.append("Слабые категории (recall<0.6): " + ", ".join(weak) + ".")
    return " ".join(parts)


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown / HTML / JSON / CSV.
# ─────────────────────────────────────────────────────────────────────────────
def render_markdown(metrics: dict[str, Any], figure_paths: dict[str, Path]) -> str:
    text = render_text_report(metrics)
    md = ["# Отчёт бенчмарка цензор-модуля\n", "```text", text, "```\n", "## Графики\n"]
    for name, p in figure_paths.items():
        md.append(f"### {name}\n")
        md.append(f"![{name}](figures/{p.name})\n")
    return "\n".join(md)


def render_html(metrics: dict[str, Any], figs: dict[str, "fig.Figure"]) -> str:
    def fig_b64(fig: "fig.Figure") -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    text = render_text_report(metrics)
    imgs = "".join(
        f"<h3>{name}</h3><img src='data:image/png;base64,{fig_b64(fig)}' style='max-width:100%'/>"
        for name, fig in figs.items()
    )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Бенчмарк цензор-модуля</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#1a1a1a}}
pre{{background:#0f1117;color:#e6e6e6;padding:16px;border-radius:8px;overflow:auto;font-size:12px}}
img{{margin:8px 0;border:1px solid #ddd;border-radius:6px}}h1{{margin-top:0}}</style></head>
<body><h1>Бенчмарк цензор-модуля</h1>
<pre>{text}</pre>
<h2>Графики</h2>{imgs}
</body></html>"""


def save_report(metrics: dict[str, Any], df: pd.DataFrame, figs: dict[str, "fig.Figure"],
                output_dir: Path, make_html: bool = True) -> Path:
    """Сохранить полный отчёт в `output_dir`. Возвращает путь к папке."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = save_figures(figs, output_dir / "figures")
    (output_dir / "report.md").write_text(render_markdown(metrics, figure_paths), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df.assign(pred_categories=df["pred_categories"].apply(lambda c: "|".join(c or []))).to_csv(
        output_dir / "predictions.csv", index=False
    )
    if make_html:
        (output_dir / "dashboard.html").write_text(render_html(metrics, figs), encoding="utf-8")
    return output_dir
