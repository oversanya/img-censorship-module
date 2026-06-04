"""Оркестратор бенчмарка: загрузка → классификация → метрики → отчёт.

Один публичный вход :func:`run_benchmark`. Держит реальный классификатор
(`ImageClassifierRunner`), сэмплит датасеты из реестра, гоняет каждую картинку
через движок с замером латентности, считает метрики и печатает/сохраняет отчёт.

Прогресс показывается в две фазы: (1) загрузка датасетов — отдельный бар на
каждый набор; (2) классификация — единый глобальный бар с ETA от tqdm. Перед
фазой 2 печатается грубая оценка времени по наблюдаемой латентности.
"""

from __future__ import annotations

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
    import os

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _warmup(runner: ImageClassifierRunner) -> float:
    """Прогрев: грузим тяжёлые модели на синтетической картинке. → секунды."""
    t0 = time.time()
    runner.classify(Image.new("RGB", (64, 64), (127, 127, 127)))
    return time.time() - t0


def _load_with_timeout(spec, n_dataset, seed, timeout, token, progress=None):
    """Загрузить датасет в daemon-потоке с anti-hang таймаутом.

    Возвращает SampleResult, либо None если воркер завис дольше `timeout` (обычно
    мёртвый сокет на чтении первого шарда). Поток — daemon, поэтому брошенный
    воркер не мешает процессу завершиться: мы идём дальше. Это НЕ ограничение на
    объём загрузки, а только страховка от бесконечного зависания.
    """
    import threading

    box: dict[str, ds_mod.SampleResult] = {}

    def worker() -> None:
        box["res"] = ds_mod.sample_dataset(
            spec, n_dataset, seed=seed, time_budget=timeout, token=token, progress=progress,
        )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return box.get("res")  # None, если поток ещё жив (завис) и не записал результат


def _estimate_load(specs, token, probe_timeout: int) -> tuple[list, list[dict[str, str]]]:
    """Фаза 0: прогрузить по 1 картинке из каждого датасета и оценить время.

    Возвращает (probed_ok, skipped), где probed_ok — список (spec, t_first) для
    источников, отдавших картинку, а skipped — список причин для остальных.
    Печатает таблицу со временем первой картинки и грубую оценку общего времени.
    """
    print(f"\n[0/2] Оценка: гружу по 1 картинке из {len(specs)} датасетов…")
    probed_ok: list[tuple[Any, float]] = []
    skipped: list[dict[str, str]] = []
    for i, spec in enumerate(specs, 1):
        bar = tqdm(total=1, desc=f"  проба {i}/{len(specs)} {spec.key[:20]:20s}", unit="img", leave=False)
        res = _load_with_timeout(spec, 1, 0, probe_timeout, token, progress=lambda k: bar.update(k))
        bar.close()
        if res is None:
            skipped.append({"dataset": spec.key, "reason": f"проба зависла >{probe_timeout}c — пропущен"})
            tqdm.write(f"  ⏱ {spec.key}: проба зависла >{probe_timeout}c")
        elif res.ok:
            t = res.load_seconds
            probed_ok.append((spec, t))
            tqdm.write(f"  ✓ {spec.key:24s} первая картинка за {t:5.1f}c")
        else:
            reason = res.skipped_reason or "пусто"
            if spec.gated_url:
                reason += f"  → примите условия: {spec.gated_url}"
            skipped.append({"dataset": spec.key, "reason": reason})
            tqdm.write(f"  ⨯ {spec.key:24s} {reason}")
    return probed_ok, skipped


def _print_estimate(probed_ok, n_dataset: int) -> None:
    """Напечатать грубую оценку общего времени загрузки и классификации."""
    if not probed_ok:
        return
    # Фикс. стоимость = открыть поток/первый шард (≈ время первой картинки).
    # Полная загрузка n картинок ≈ эта стоимость + докачка остальных. Берём
    # консервативный коридор [×1, ×3] от суммы времён первой картинки.
    sum_first = sum(t for _, t in probed_ok)
    lo, hi = sum_first, sum_first * 3.0
    est_imgs = n_dataset * len(probed_ok)  # верхняя оценка (per_class даёт больше)
    cls_s = est_imgs * 0.05
    print(f"\n  Оценка загрузки ~{lo:.0f}–{hi:.0f} c на {len(probed_ok)} датасетов "
          f"(зависит от сети). Классификация ~{cls_s:.0f} c на ≈{est_imgs} картинок.")
    print(f"  Итого ориентировочно ~{lo + cls_s:.0f}–{hi + cls_s:.0f} c. Гружу всё доступное…")


def run_benchmark(
    n_dataset: int = 100,
    save_report: bool = True,
    output_dir: str | Path = "reports",
    datasets: list[str] | None = None,
    seed: int = 42,
    per_dataset_timeout: int = 1200,
    probe_timeout: int = 90,
    make_html: bool = True,
    print_report: bool = True,
) -> dict[str, Any]:
    """Прогнать бенчмарк и вернуть пакет метрик.

    Грузит ВСЁ, что грузится: недоступные источники (gated без принятых условий,
    мёртвые id, несовместимые схемы) просто пропускаются. Жёстких лимитов на
    объём загрузки нет — только anti-hang таймаут от зависшей сети.

    Args:
        n_dataset: сколько картинок брать из каждого датасета. Для UnsafeBench
            (per_class) — до `n_dataset` на КАЖДЫЙ класс (категория × safe/unsafe).
        save_report: сохранять ли артефакты (md/png/csv/json[/html]).
        output_dir: куда складывать `benchmark_<timestamp>/`.
        datasets: список ключей/категорий для фильтра реестра (None → все).
        seed: сид сэмплирования.
        per_dataset_timeout: страховка от зависания сети на одном датасете (с).
            Это не ограничение объёма — обычная загрузка завершается куда раньше.
        probe_timeout: таймаут пробной загрузки 1 картинки в фазе оценки (с).
        make_html: генерировать ли HTML-дашборд (при save_report).
        print_report: печатать ли финальный отчёт в консоль.

    Returns:
        dict с ключами config / latency / overall / per_category /
        per_dataset / adversarial / skipped_datasets / report_dir / dataframe.
    """
    specs = ds_mod.select_specs(datasets)
    token = _resolve_token()

    runner = ImageClassifierRunner()
    print(f"Прогрев моделей… (block≥{runner.settings.block_threshold}, review≥{runner.settings.review_threshold})")
    warmup_s = _warmup(runner)
    print(f"Модели загружены за {warmup_s:.1f} c. HF_TOKEN: {'есть' if token else 'нет (gated пропустятся)'}")

    # ── Фаза 0: оценка времени (по 1 картинке из каждого датасета) ───────────
    probed_ok, skipped = _estimate_load(specs, token, probe_timeout)
    _print_estimate(probed_ok, n_dataset)
    if not probed_ok:
        raise RuntimeError("Ни один датасет не отдал картинку — нечего оценивать. Проверьте сеть/HF_TOKEN.")

    # ── Фаза 1: полная загрузка (только то, что прошло пробу) ─────────────────
    print(f"\n[1/2] Загрузка {len(probed_ok)} датасетов (до {n_dataset} картинок"
          f"{' на класс' if any(s.per_class for s, _ in probed_ok) else ''})…")
    samples: list[ds_mod.SampleResult] = []
    for i, (spec, _) in enumerate(probed_ok, 1):
        # per_class набирает до n на каждый класс → итоговая цель кратно больше.
        bar_total = n_dataset * 12 if spec.per_class else n_dataset
        bar = tqdm(total=bar_total, desc=f"  {i}/{len(probed_ok)} {spec.key[:20]:20s}",
                   unit="img", leave=False)
        res = _load_with_timeout(spec, n_dataset, seed, per_dataset_timeout, token,
                                 progress=lambda k: bar.update(k))
        bar.close()
        if res is None:
            skipped.append({"dataset": spec.key, "reason": f"загрузка зависла >{per_dataset_timeout}c"})
            tqdm.write(f"  ⏱ {i}/{len(probed_ok)} {spec.key}: зависла >{per_dataset_timeout}c — пропускаю")
        elif res.ok:
            samples.append(res)
            tqdm.write(f"  ✓ {i}/{len(probed_ok)} {spec.key}: {len(res.items)} картинок за {res.load_seconds:.0f}c")
        else:
            skipped.append({"dataset": spec.key, "reason": res.skipped_reason or "пусто"})
            tqdm.write(f"  ⨯ {i}/{len(probed_ok)} {spec.key}: {res.skipped_reason}")

    total_imgs = sum(len(s.items) for s in samples)
    if total_imgs == 0:
        raise RuntimeError("Ни один датасет не загрузился — нечего оценивать. Проверьте сеть/HF_TOKEN.")

    # ── Фаза 2: классификация ────────────────────────────────────────────────
    print(f"\n[2/2] Классификация {total_imgs} изображений из {len(samples)} датасетов…")
    rows: list[dict[str, Any]] = []
    codes = [spec.code for spec in CATEGORY_SPECS]
    bar = tqdm(total=total_imgs, desc="classify", unit="img")
    for sample in samples:
        for image, label in sample.items:
            t0 = time.time()
            result = runner.classify(image)
            dt = time.time() - t0
            row: dict[str, Any] = {
                "dataset": sample.spec.key,
                "dataset_category": sample.spec.category,
                "true_unsafe": label.true_unsafe,
                "true_category": label.true_category,
                "adversarial": label.adversarial,
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
            bar.update(1)
            done = len(rows)
            if done % 25 == 0:
                mean_lat = bar.format_dict["elapsed"] / max(1, done)
                bar.set_postfix_str(f"~{mean_lat*1000:.0f}ms/img, осталось ≈{mean_lat*(total_imgs-done):.0f}c")
    bar.close()

    df = pd.DataFrame(rows)
    # true_unsafe → nullable boolean (могут быть None).
    df["true_unsafe"] = df["true_unsafe"].astype("object")
    wall_s = float(df["latency_s"].sum())

    # ── Метрики ──────────────────────────────────────────────────────────────
    config = {
        "n_dataset": n_dataset,
        "n_datasets_used": len(samples),
        "datasets_used": [s.spec.key for s in samples],
        "seed": seed,
        "block_threshold": runner.settings.block_threshold,
        "review_threshold": runner.settings.review_threshold,
        "visual_model_id": runner.settings.visual_model_id,
        "explicit_model_id": runner.settings.explicit_model_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    metrics = metrics_mod.compute_all(
        df, warmup_seconds=warmup_s, wall_seconds=wall_s, config=config, skipped=skipped
    )

    # ── Отчёт ────────────────────────────────────────────────────────────────
    figs = report_mod.build_figures(
        df, metrics, runner.settings.block_threshold, runner.settings.review_threshold
    )
    report_dir = None
    if save_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(output_dir) / f"benchmark_{stamp}"
        report_mod.save_report(metrics, df, figs, report_dir, make_html=make_html)

    if print_report:
        print("\n" + report_mod.render_text_report(metrics))
        if report_dir is not None:
            print(f"\nОтчёт сохранён: {report_dir}/  (report.md, dashboard.html, predictions.csv, metrics.json, figures/)")

    metrics["report_dir"] = str(report_dir) if report_dir else None
    metrics["dataframe"] = df
    metrics["figures"] = figs
    return metrics
