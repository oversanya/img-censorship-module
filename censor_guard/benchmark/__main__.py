"""CLI: запуск бенчмарка одной командой.

    python -m censor_guard.benchmark --n-dataset 100 --save-report

Аргументы повторяют параметры :func:`censor_guard.benchmark.run_benchmark`.
"""

from __future__ import annotations

import argparse

from censor_guard.benchmark.core import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m censor_guard.benchmark",
        description="Бенчмарк image-классификатора цензор-модуля по датасетам из censor_benchmark_datasets.md.",
    )
    p.add_argument("--n-dataset", type=int, default=100, help="картинок на датасет (по умолчанию 100)")
    save = p.add_mutually_exclusive_group()
    save.add_argument("--save-report", dest="save_report", action="store_true", help="сохранять отчёт (по умолчанию)")
    save.add_argument("--no-save-report", dest="save_report", action="store_false", help="не сохранять отчёт")
    p.set_defaults(save_report=True)
    p.add_argument("--output-dir", default="reports", help="куда сохранять отчёты (по умолчанию ./reports)")
    p.add_argument("--datasets", default=None,
                   help="фильтр: csv из ключей датасетов ИЛИ кодов категорий (по умолчанию все)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-dataset-timeout", type=int, default=1200,
                   help="anti-hang таймаут на датасет в секундах (страховка от зависшей сети, не лимит объёма)")
    p.add_argument("--probe-timeout", type=int, default=90,
                   help="таймаут пробной загрузки 1 картинки в фазе оценки (с)")
    p.add_argument("--no-html", dest="make_html", action="store_false", help="не генерировать HTML-дашборд")
    p.set_defaults(make_html=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    datasets = [d for d in args.datasets.split(",")] if args.datasets else None
    run_benchmark(
        n_dataset=args.n_dataset,
        save_report=args.save_report,
        output_dir=args.output_dir,
        datasets=datasets,
        seed=args.seed,
        per_dataset_timeout=args.per_dataset_timeout,
        probe_timeout=args.probe_timeout,
        make_html=args.make_html,
    )
    # Гарантируем немедленный выход CLI: библиотека `datasets` может оставлять
    # фоновые потоки, из-за которых обычный возврат «висит». Отчёт уже сохранён
    # и напечатан, поэтому принудительный выход безопасен.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
