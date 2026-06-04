"""Бенчмарк-харнесс для image-классификатора цензор-модуля.

Один публичный вход — :func:`run_benchmark`. Гоняет реальный классификатор
(`censor_guard.evaluation.ImageClassifierRunner`) по набору датасетов из
`censor_benchmark_datasets.md`, считает метрики (общие, по категориям таксономии
и по датасетам), строит графики и печатает/сохраняет подробный отчёт.

Запуск одной командой:

    python -m censor_guard.benchmark --n-dataset 100 --save-report

Или программно:

    from censor_guard.benchmark import run_benchmark
    result = run_benchmark(n_dataset=100, save_report=True)
"""

from __future__ import annotations

from censor_guard.benchmark.core import run_benchmark
from censor_guard.benchmark.report import show_figures

__all__ = ["run_benchmark", "show_figures"]
