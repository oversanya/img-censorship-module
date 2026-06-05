"""Интерактивные графики (Plotly), консольный/Markdown-отчёт и HTML-дашборд.

Графики строятся на Plotly и встраиваются инлайн в один самодостаточный
`dashboard.html` (plotly.js встроен один раз → открывается без сервера и без сети).
Те же объекты `go.Figure` возвращаются, чтобы ноутбук показал их интерактивно.
Консольный/Markdown-отчёт текстовый — для CLI и быстрого взгляда.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from censor_guard.benchmark.metrics import _clean_mask, category_verdict

# Палитра, общая для всех графиков.
C_SAFE, C_UNSAFE, C_BLUE, C_PURPLE, C_TEAL = "#2e7d32", "#c62828", "#1565c0", "#6a1b9a", "#00838f"
_TEMPLATE = "plotly_white"


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return "н/д" if math.isnan(x) else f"{x:.3f}"
    return str(x)


def _nan0(x: Any) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _ascii_table(headers: list[str], rows: list[list[Any]]) -> str:
    cols = [headers] + [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(str(cols[r][c])) for r in range(len(cols))) for c in range(len(headers))]
    line = "─┼─".join("─" * w for w in widths)
    out = [" │ ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)), line]
    for r in rows:
        out.append(" │ ".join(_fmt(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Графики (Plotly).
# ─────────────────────────────────────────────────────────────────────────────
def build_figures(df: pd.DataFrame, metrics: dict[str, Any], block_thr: float, review_thr: float) -> dict[str, go.Figure]:
    figs: dict[str, go.Figure] = {}
    clean = df[_clean_mask(df)]

    # (1) Confusion matrix + распределение unsafe_score.
    if not clean.empty:
        from sklearn.metrics import confusion_matrix

        f = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                          subplot_titles=("Confusion matrix (Safe vs Unsafe)", "Распределение unsafe_score"))
        cm = confusion_matrix(clean["true_unsafe"].astype(bool), clean["flagged"].astype(bool), labels=[False, True])
        f.add_trace(go.Heatmap(
            z=cm, x=["pred Safe", "pred Unsafe"], y=["true Safe", "true Unsafe"],
            text=cm, texttemplate="%{text}", textfont={"size": 16}, colorscale="Blues",
            showscale=False, hovertemplate="%{y} → %{x}: %{z}<extra></extra>"), row=1, col=1)
        for val, color, name in [(False, C_SAFE, "Safe"), (True, C_UNSAFE, "Unsafe")]:
            f.add_trace(go.Histogram(
                x=clean[clean["true_unsafe"] == val]["unsafe_score"], name=name,
                marker_color=color, opacity=0.6, xbins={"start": 0, "end": 1, "size": 0.05}), row=1, col=2)
        f.add_vline(x=block_thr, line_dash="dash", line_color="black", annotation_text=f"block {block_thr}", row=1, col=2)
        f.add_vline(x=review_thr, line_dash="dot", line_color="grey", annotation_text=f"review {review_thr}", row=1, col=2)
        f.update_layout(barmode="overlay", template=_TEMPLATE, height=400,
                        legend={"x": 0.99, "y": 0.99, "xanchor": "right"})
        f.update_xaxes(title_text="unsafe_score", row=1, col=2)
        figs["overview"] = f

    # (2) ROC overall + overlay по категориям.
    if not clean.empty and clean["true_unsafe"].nunique() > 1:
        from sklearn.metrics import roc_curve

        f = go.Figure()
        fpr, tpr, _ = roc_curve(clean["true_unsafe"].astype(bool), clean["unsafe_score"])
        auc = metrics["overall"].get("roc_auc")
        f.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"overall (AUC={_fmt(auc)})",
                               line={"color": C_BLUE, "width": 3}))
        negatives = clean[clean["true_unsafe"] == False]  # noqa: E712
        for code, cm in metrics["per_category"].items():
            if not cm.get("supported"):
                continue
            pos = clean[(clean["true_unsafe"] == True) & (clean["true_category"] == code)]  # noqa: E712
            pool = pd.concat([pos, negatives], ignore_index=True)
            if pool["true_unsafe"].nunique() < 2:
                continue
            fc, tc, _ = roc_curve((pool["true_unsafe"] == True).astype(bool), pool[f"score_{code}"])  # noqa: E712
            f.add_trace(go.Scatter(x=fc, y=tc, mode="lines", name=f"{code} ({_fmt(cm.get('roc_auc'))})",
                                   line={"width": 1.3}, opacity=0.75))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="случайный",
                               line={"color": "grey", "dash": "dash"}, showlegend=False))
        f.update_layout(title="ROC: общий и по категориям", xaxis_title="FPR", yaxis_title="TPR",
                        template=_TEMPLATE, height=520, legend={"font": {"size": 10}})
        figs["roc"] = f

    # (3) Метрики по категориям таксономии.
    cats = [(c, m) for c, m in metrics["per_category"].items() if m.get("supported")]
    if cats:
        labels = [c for c, _ in cats]
        f = go.Figure()
        for key, color in [("recall", C_UNSAFE), ("precision", C_BLUE), ("f1", C_SAFE)]:
            f.add_trace(go.Bar(name=key, x=labels, y=[_nan0(m.get(key)) for _, m in cats],
                               marker_color=color, opacity=0.88,
                               hovertemplate="%{x}<br>" + key + "=%{y:.3f}<extra></extra>"))
        f.update_layout(barmode="group", title="Метрики по категориям таксономии",
                        yaxis={"range": [0, 1], "title": "score"}, template=_TEMPLATE, height=440)
        figs["per_category"] = f

    # (4) По источникам: recall / fpr.
    pdm = metrics["per_dataset"]
    if pdm:
        names = [r["dataset"] for r in pdm]
        f = go.Figure()
        f.add_trace(go.Bar(name="recall", x=names, y=[_nan0(r.get("recall")) for r in pdm], marker_color=C_PURPLE))
        f.add_trace(go.Bar(name="FPR", x=names, y=[_nan0(r.get("fpr")) for r in pdm], marker_color=C_UNSAFE))
        f.update_layout(barmode="group", title="Recall / FPR по источникам данных",
                        yaxis={"range": [0, 1]}, template=_TEMPLATE, height=440)
        figs["by_source"] = f

    # (5) AI-генерация vs реальные.
    sl = metrics.get("by_ai_generated") or []
    if sl:
        names = ["AI-генерация" if r["ai_generated"] in ("True", "1") else "Реальные" for r in sl]
        f = go.Figure()
        for key, color in [("recall", C_UNSAFE), ("precision", C_BLUE), ("fpr", "#ef6c00")]:
            f.add_trace(go.Bar(name=key, x=names, y=[_nan0(r.get(key)) for r in sl], marker_color=color, opacity=0.88))
        f.update_layout(barmode="group", title="Качество: AI-генерация vs реальные изображения",
                        yaxis={"range": [0, 1]}, template=_TEMPLATE, height=400)
        figs["by_ai"] = f

    # (6) Латентность.
    lat = df["latency_s"].to_numpy(dtype=float) * 1000
    lat = lat[lat > 0]
    if lat.size:
        f = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                          subplot_titles=("Распределение latency", "Boxplot latency"))
        f.add_trace(go.Histogram(x=lat, nbinsx=30, marker_color=C_TEAL, showlegend=False), row=1, col=1)
        f.add_trace(go.Box(x=lat, marker_color=C_TEAL, boxpoints=False, showlegend=False), row=1, col=2)
        f.update_xaxes(title_text="latency, мс")
        f.update_layout(template=_TEMPLATE, height=360)
        figs["latency"] = f

    return figs


def show_figures(figs: dict[str, go.Figure]) -> None:
    """Показать интерактивные графики inline в Jupyter."""
    for fig in figs.values():
        fig.show()


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
    add(f"Датасет             : {cfg.get('dataset_id', '—')} [{cfg.get('split', '—')}]")
    add(f"Всего изображений   : {metrics['n_images']}")
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

    # Общие метрики (на «чистом» наборе).
    add("─ ОБЩИЕ МЕТРИКИ (Safe vs Unsafe, без adversarial/edge) " + "─" * 23)
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
        rows.append([code, m["n_positive"], m["recall"], m["precision"], m["fpr"], m["roc_auc"],
                     category_verdict(m["recall"], m.get("fpr"))])
    add(_ascii_table(["категория", "n+", "recall", "precision", "FPR", "ROC-AUC", "оценка"], rows))
    add("")

    # По источникам.
    add("─ ПО ИСТОЧНИКАМ ДАННЫХ " + "─" * 55)
    rows = [[r["dataset"], r["n"], r.get("recall", "—"), r.get("precision", "—"),
             r.get("fpr", "—"), r.get("roc_auc", "—"), r["mean_latency_ms"]]
            for r in metrics["per_dataset"]]
    add(_ascii_table(["источник", "n", "recall", "precision", "FPR", "ROC-AUC", "lat,мс"], rows))
    add("")

    # Срез по AI-генерации.
    sl = metrics.get("by_ai_generated") or []
    if sl:
        add("─ AI-ГЕНЕРАЦИЯ vs РЕАЛЬНЫЕ " + "─" * 51)
        rows = [["AI" if r["ai_generated"] in ("True", "1") else "реальные", r["n"],
                 r.get("recall", "—"), r.get("precision", "—"), r.get("fpr", "—"), r.get("roc_auc", "—")] for r in sl]
        add(_ascii_table(["тип", "n", "recall", "precision", "FPR", "ROC-AUC"], rows))
        add("")

    # Robustness + edge-case.
    adv = metrics["adversarial"]
    edge = metrics.get("edge_case", {})
    if adv.get("n") or edge.get("n"):
        add("─ УСТОЙЧИВОСТЬ (ADVERSARIAL / EDGE-CASE) " + "─" * 37)
        if adv.get("obfuscated_recall") is not None:
            add(f"Обфусцированный unsafe (ожидаем block): recall = {adv['obfuscated_recall']} "
                f"на {adv['n_obfuscated']} картинках")
        if adv.get("hard_negative_fpr") is not None:
            add(f"Adversarial hard-negatives (ожидаем allow): FPR = {adv['hard_negative_fpr']} "
                f"на {adv['n_hard_negative']} картинках")
        if edge.get("fpr") is not None:
            add(f"Edge-case hard-negatives (ожидаем allow): FPR = {edge['fpr']} "
                f"({edge['n_false_block']}/{edge['n_safe']} ложных блокировок)")
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
    recall, fpr, auc = overall["recall"], overall["fpr"], overall["roc_auc"]
    parts = [f"Общий разделяющий потенциал ROC-AUC={_fmt(auc)}, recall={_fmt(recall)}, FPR={_fmt(fpr)}."]
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
# HTML-дашборд (single-file, plotly.js встроен инлайн).
# ─────────────────────────────────────────────────────────────────────────────
def _figure_divs(figs: dict[str, go.Figure]) -> dict[str, str]:
    """Каждый график → <div>. plotly.js встраиваем инлайн только в первый."""
    divs: dict[str, str] = {}
    for i, (name, fig) in enumerate(figs.items()):
        divs[name] = fig.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False),
                                 default_width="100%", config={"displaylogo": False})
    return divs


def _card(value: Any, label: str, good: bool | None = None) -> str:
    cls = "" if good is None else (" good" if good else " bad")
    return f'<div class="card{cls}"><div class="val">{_fmt(value)}</div><div class="lbl">{label}</div></div>'


def _cat_table_html(metrics: dict[str, Any]) -> str:
    head = "".join(f"<th>{h}</th>" for h in
                   ["категория", "n+", "recall", "precision", "f1", "FPR", "ROC-AUC", "оценка"])
    body = []
    for code, m in metrics["per_category"].items():
        if not m.get("supported"):
            body.append(f"<tr class='dim'><td>{code}</td><td>{m.get('n_positive', 0)}</td>"
                        f"<td colspan='5'>—</td><td>нет данных</td></tr>")
            continue
        v = category_verdict(m["recall"], m.get("fpr"))
        cls = "ok" if m["recall"] >= 0.85 else ("warn" if m["recall"] >= 0.6 else "bad")
        body.append(
            f"<tr><td>{code}</td><td>{m['n_positive']}</td><td class='{cls}'>{_fmt(m['recall'])}</td>"
            f"<td>{_fmt(m['precision'])}</td><td>{_fmt(m['f1'])}</td><td>{_fmt(m['fpr'])}</td>"
            f"<td>{_fmt(m['roc_auc'])}</td><td>{v}</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(metrics: dict[str, Any], figs: dict[str, go.Figure]) -> str:
    cfg, overall, lat = metrics["config"], metrics["overall"], metrics["latency"]
    div = _figure_divs(figs)
    g = lambda name: f"<div class='chart'>{div.get(name, '')}</div>" if name in div else ""  # noqa: E731

    cards = "".join([
        _card(overall.get("recall"), "Recall (Unsafe)", good=_nan0(overall.get("recall")) >= 0.85),
        _card(overall.get("fpr"), "FPR (ложные блокировки)", good=_nan0(overall.get("fpr")) <= 0.15),
        _card(overall.get("precision"), "Precision"),
        _card(overall.get("f1"), "F1"),
        _card(overall.get("roc_auc"), "ROC-AUC", good=_nan0(overall.get("roc_auc")) >= 0.85),
        _card(metrics["n_images"], "Изображений"),
        _card(lat.get("mean_ms"), "Latency mean, мс"),
        _card(lat.get("throughput_img_s"), "img/s"),
    ])

    adv, edge = metrics["adversarial"], metrics.get("edge_case", {})
    robust = "".join([
        _card(adv.get("obfuscated_recall"), f"Recall на обфускациях (n={adv.get('n_obfuscated', 0)})",
              good=_nan0(adv.get("obfuscated_recall")) >= 0.7) if adv.get("obfuscated_recall") is not None else "",
        _card(edge.get("fpr"), f"FPR на edge-case (n={edge.get('n_safe', 0)})",
              good=_nan0(edge.get("fpr")) <= 0.2) if edge.get("fpr") is not None else "",
    ])
    robust_block = f"<h2>Устойчивость: adversarial & edge-case</h2><div class='cards'>{robust}</div>" if robust else ""

    assessment = _overall_assessment(metrics)
    cfg_rows = "".join(f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>" for k, v in cfg.items())

    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Бенчмарк цензор-модуля</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--mut:#6b7280;--ok:#2e7d32;--bad:#c62828;--warn:#ef6c00;--bd:#e5e7eb}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:var(--bg);color:var(--ink)}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}}
h1{{margin:0 0 4px}}h2{{margin:34px 0 14px;font-size:20px}}
.sub{{color:var(--mut);margin-bottom:22px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}}
.card .val{{font-size:26px;font-weight:700}}.card .lbl{{color:var(--mut);font-size:13px;margin-top:4px}}
.card.good .val{{color:var(--ok)}}.card.bad .val{{color:var(--bad)}}
.chart{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:8px;margin:14px 0}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:12px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--bd);font-size:14px}}
th{{background:#fafbfc;font-weight:600}}tr.dim td{{color:var(--mut)}}
td.ok{{color:var(--ok);font-weight:600}}td.warn{{color:var(--warn);font-weight:600}}td.bad{{color:var(--bad);font-weight:600}}
.note{{background:#eef3fb;border-left:4px solid var(--blue,#1565c0);border-radius:8px;padding:14px 16px;line-height:1.55}}
details{{margin-top:30px}}summary{{cursor:pointer;color:var(--mut)}}
</style></head>
<body><div class="wrap">
<h1>Бенчмарк цензор-модуля</h1>
<div class="sub">{cfg.get('dataset_id', '')} · split <b>{cfg.get('split', '')}</b> · {metrics['n_images']} изображений · {cfg.get('timestamp', '')}</div>

<div class="cards">{cards}</div>
<div class="note" style="margin-top:18px">{assessment}</div>

<h2>Обзор: матрица ошибок и распределение оценок</h2>{g('overview')}
<h2>ROC: общий и по категориям</h2>{g('roc')}
<h2>Метрики по категориям таксономии</h2>{_cat_table_html(metrics)}{g('per_category')}
<h2>По источникам данных</h2>{g('by_source')}
<h2>AI-генерация vs реальные изображения</h2>{g('by_ai')}
{robust_block}
<h2>Латентность</h2>{g('latency')}

<details><summary>Конфигурация прогона</summary>
<table style="margin-top:10px"><tbody>{cfg_rows}</tbody></table></details>
</div></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Сохранение артефактов.
# ─────────────────────────────────────────────────────────────────────────────
def render_markdown(metrics: dict[str, Any]) -> str:
    return "# Отчёт бенчмарка цензор-модуля\n\n```text\n" + render_text_report(metrics) + \
        "\n```\n\nИнтерактивные графики — в `dashboard.html`.\n"


def save_report(metrics: dict[str, Any], df: pd.DataFrame, figs: dict[str, go.Figure],
                output_dir: Path, make_html: bool = True) -> Path:
    """Сохранить отчёт в `output_dir`. Возвращает путь к папке."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(render_markdown(metrics), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k not in ("dataframe", "figures")},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df.assign(pred_categories=df["pred_categories"].apply(lambda c: "|".join(c or []))).to_csv(
        output_dir / "predictions.csv", index=False)
    if make_html:
        (output_dir / "dashboard.html").write_text(render_html(metrics, figs), encoding="utf-8")
    return output_dir
