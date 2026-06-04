from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibratedScores:
    """Результат калибровки zero-shot выхода относительно safe-якоря."""

    # {код категории: откалиброванная оценка 0..1}
    scores: dict[str, float]
    # Максимальная softmax-оценка среди безопасных якорей — «база», с которой
    # сравнивались категории нарушений.
    safe_score: float
    # Параметр калибровки (порог равных шансов с safe), с которым считали.
    floor: float


def calibrate_against_safe(
    raw_scores: dict[str, float],
    label_to_code: dict[str, str],
    safe_labels: set[str],
    floor: float = 0.5,
) -> CalibratedScores:
    """Превращает «относительное сходство» zero-shot CLIP в честную оценку опасности.

    Проблема сырого zero-shot: softmax по всем меткам в сумме даёт ~1.0, поэтому у
    любой картинки (даже котика) всегда есть «самая вероятная» категория нарушения.
    Сравнивать такие оценки с абсолютным порогом нельзя.

    Идея калибровки: для каждой категории c берём её сырую оценку s_c и оценку
    safe-якоря s_safe (максимум среди нескольких нейтральных формулировок) и считаем
    бинарную вероятность «скорее c, чем safe»::

        p = s_c / (s_c + s_safe)

    p ≈ 0.5 означает «модель не уверена, что это опаснее, чем нейтральная картинка» —
    такой сигнал нам не нужен. Поэтому оставляем только уверенность ВЫШЕ равных
    шансов с safe и линейно растягиваем её в [0, 1]::

        calibrated = clamp((p - floor) / (1 - floor), 0, 1)

    Итог: безобидная картинка (категория едва обгоняет safe) → ~0; картинка, где
    категория уверенно доминирует над safe → ~1. Теперь оценку можно сравнивать с
    порогами и сливать с другими сенсорами на одной шкале.
    """

    safe_score = max((raw_scores.get(label, 0.0) for label in safe_labels), default=0.0)
    denom = max(1e-6, 1.0 - floor)

    scores: dict[str, float] = {}
    for label, code in label_to_code.items():
        raw = raw_scores.get(label)
        if raw is None:
            continue
        total = raw + safe_score
        if total <= 0.0:
            scores[code] = 0.0
            continue
        # Если safe-якорь не нашёлся вовсе (нетипично), деградируем к сырой оценке,
        # а не выдаём ложную единицу.
        p = raw / total if safe_score > 0.0 else raw
        calibrated = (p - floor) / denom
        scores[code] = min(1.0, max(0.0, calibrated))

    return CalibratedScores(scores=scores, safe_score=safe_score, floor=floor)
