"""Бенчмарк-харнесс для image-классификатора цензор-модуля.

Один публичный вход — :func:`run_benchmark`. Гоняет реальный классификатор
(`censor_guard.evaluation.ImageClassifierRunner` = CLIP zero-shot + NSFW-детектор)
по курируемому датасету `vekshinkir/image-censorship-small` (split `benchmarking`),
считает метрики (общие, по категориям таксономии, по источникам, AI-vs-реальные,
adversarial/edge-case), строит интерактивные Plotly-графики и собирает
самодостаточный HTML-дашборд.

Запуск одной командой:

    python -m censor_guard.benchmark --limit 60

Или программно:

    from censor_guard.benchmark import run_benchmark, show_figures
    result = run_benchmark(split="benchmarking")
    show_figures(result["figures"])
"""

from __future__ import annotations

from censor_guard.benchmark.core import run_benchmark
from censor_guard.benchmark.report import show_figures

__all__ = ["run_benchmark", "show_figures"]
