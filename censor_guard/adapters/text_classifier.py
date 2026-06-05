from __future__ import annotations

import os
from pathlib import Path

from censor_guard.schemas import SignalResult
from censor_guard.taxonomy import TEXT_LEXICONS


# Метки, означающие «безопасно» — их НЕ маппим ни в какую категорию. Проверяются
# до основного маппинга (важно: "non-toxic" содержит подстроку "toxic", иначе
# безопасный текст ложно улетел бы в harassment).
SAFE_LABEL_MARKERS: tuple[str, ...] = (
    "non-toxic", "non_toxic", "not toxic", "neutral", "normal", "safe", "clean", "ok",
)

# Маркеры меток внешних text-classification моделей → коды нашей таксономии.
# Сверяются подстрокой по label.lower(), поэтому ловят разные форматы
# ("toxic", "S_HATE", "sexual_explicit", "insult" и т.п.). Используется только если
# подключена ML-модель (model_id задан). Дефолт — cointegrated/rubert-tiny-toxicity
# (метки insult/obscenity/threat/dangerous/non-toxic):
#   insult, obscenity → harassment;  threat → violence_gore.
# "dangerous" намеренно НЕ маплем — у этой модели он шумит почти на любом токсичном
# тексте и давал бы ложный illegal_activity.
ML_LABEL_MARKERS: tuple[tuple[str, str], ...] = (
    ("sexual", "sexual"),
    ("porn", "sexual"),
    ("obscenity", "harassment"),
    ("violence", "violence_gore"),
    ("threat", "violence_gore"),
    ("self-harm", "self_harm"),
    ("self_harm", "self_harm"),
    ("suicide", "self_harm"),
    ("hate", "hate_extremism"),
    ("harass", "harassment"),
    ("insult", "harassment"),
    ("toxic", "harassment"),
)


def _normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


class TextGuard:
    """Текстовый гард для промпта и OCR-текста.

    Базовый слой — лексиконный классификатор (без ML, всегда доступен): ищет в
    тексте стемы запрещённых тем (taxonomy.TEXT_LEXICONS, ru+en) и выдаёт
    откалиброванную оценку по категориям. Это осознанно простой и объяснимый
    baseline — он заменяет прежнюю заглушку (которая считала любой текст
    безопасным), но не претендует на полноту ML-модели.

    Опциональный слой — внешняя HF text-classification модель (если задан model_id
    и установлен transformers): её оценки маппятся в таксономию по ML_LABEL_MARKERS
    и сливаются с лексиконом по максимуму. При недоступности модели тихо
    деградируем к одному лексикону.
    """

    name = "text_guard"

    def __init__(
        self,
        enabled: bool = True,
        model_id: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.model_id = model_id or None
        self.cache_dir = cache_dir
        self._pipeline = None
        self._load_error: str | None = None

    # --- лексиконный слой -------------------------------------------------
    def _lexicon_scores(self, text_norm: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores: dict[str, float] = {}
        matches: dict[str, list[str]] = {}
        for code, stems in TEXT_LEXICONS.items():
            hit = [stem for stem in stems if stem in text_norm]
            if not hit:
                continue
            # Один специфичный термин — уже сигнал (но обычно review, не блок);
            # несколько совпадений усиливают уверенность вплоть до блокировки.
            score = min(0.95, 0.6 + 0.12 * len(hit))
            scores[code] = score
            matches[code] = hit
        return scores, matches

    # --- опциональный ML-слой --------------------------------------------
    def _load_ml(self):
        if self._pipeline is not None or self._load_error is not None or not self.model_id:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError:
            return None
        try:
            if self.cache_dir:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", self.cache_dir)
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
            # sigmoid: toxicity-модели мультилейбловые (метки независимы, не softmax).
            self._pipeline = pipeline(
                task="text-classification",
                model=self.model_id,
                top_k=None,
                function_to_apply="sigmoid",
            )
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = str(exc)
            return None
        return self._pipeline

    def _ml_scores(self, text: str) -> tuple[dict[str, float], dict]:
        pipe = self._load_ml()
        if pipe is None:
            return {}, {}
        try:
            raw = pipe(text)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return {}, {"ml_error": str(exc)}
        # transformers с top_k=None возвращает список (или список списков) пар label/score.
        rows = raw[0] if raw and isinstance(raw[0], list) else raw
        scores: dict[str, float] = {}
        raw_map: dict[str, float] = {}
        for item in rows:
            label = str(item["label"]).lower()
            score = float(item["score"])
            raw_map[label] = score
            if any(safe in label for safe in SAFE_LABEL_MARKERS):
                continue  # «non-toxic»/«neutral»/... не маппим в нарушения
            if score < 0.05:
                continue  # отсекаем шум, чтобы не плодить нулевые категории
            for marker, code in ML_LABEL_MARKERS:
                if marker in label:
                    scores[code] = max(scores.get(code, 0.0), score)
        return scores, {"ml_raw": raw_map}

    def moderate(self, text: str | None) -> SignalResult:
        if not self.enabled:
            return SignalResult(name=self.name, status="skipped", reason="Text guard disabled by configuration.")
        if not text or not text.strip():
            return SignalResult(name=self.name, status="skipped", reason="No text supplied.")

        text_norm = _normalize(text)
        lex_scores, lex_matches = self._lexicon_scores(text_norm)
        ml_scores, ml_raw = self._ml_scores(text)

        fused: dict[str, float] = dict(lex_scores)
        for code, score in ml_scores.items():
            fused[code] = max(fused.get(code, 0.0), score)

        reason = (
            "No flagged terms in text."
            if not fused
            else f"Flagged {len(fused)} categor{'y' if len(fused) == 1 else 'ies'} from text."
        )
        return SignalResult(
            name=self.name,
            status="ok",
            categories=fused,
            reason=reason,
            raw={
                "mode": "lexicon+ml" if self.model_id else "lexicon",
                "lexicon_matches": lex_matches,
                "text_length": len(text),
                **ml_raw,
            },
        )
