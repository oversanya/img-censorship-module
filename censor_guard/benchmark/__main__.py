"""CLI: запуск бенчмарка одной командой.

    python -m censor_guard.benchmark                 # весь split benchmarking
    python -m censor_guard.benchmark --limit 60      # быстрая проба
"""

from __future__ import annotations

import argparse

from censor_guard.benchmark import datasets as ds_mod
from censor_guard.benchmark.core import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m censor_guard.benchmark",
        description=f"Бенчмарк image-классификатора цензор-модуля на {ds_mod.DATASET_ID}.",
    )
    p.add_argument("--split", default=ds_mod.DEFAULT_SPLIT, help=f"сплит датасета (по умолчанию {ds_mod.DEFAULT_SPLIT})")
    p.add_argument("--limit", type=int, default=None, help="ограничить число картинок (по умолчанию все)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="reports", help="куда сохранять отчёты (по умолчанию ./reports)")
    p.add_argument("--no-save-report", dest="save_report", action="store_false", help="не сохранять отчёт")
    p.add_argument("--no-html", dest="make_html", action="store_false", help="не генерировать HTML-дашборд")
    p.set_defaults(save_report=True, make_html=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        save_report=args.save_report,
        output_dir=args.output_dir,
        make_html=args.make_html,
    )
    # Гарантируем немедленный выход CLI: библиотека `datasets` может оставлять
    # фоновые потоки. Отчёт уже сохранён и напечатан — принудительный выход безопасен.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
