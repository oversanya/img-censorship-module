from __future__ import annotations

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.ocr import OCRAdapter
from censor_guard.adapters.policy_judge import PolicyJudge
from censor_guard.adapters.text_classifier import TextGuardHeuristic
from censor_guard.adapters.visual_classifier import VisualClassifierAdapter
from censor_guard.config import Settings
from censor_guard.decision import DecisionEngine
from censor_guard.image_utils import load_image
from censor_guard.schemas import ModerationRequest, ModerationResponse


class GuardrailPipeline:
    """Оркестратор всего конвейера модерации.

    Создаёт все адаптеры-«сенсоры» один раз (в __init__) и переиспользует их
    между запросами. Тяжёлые ML-модели внутри адаптеров грузятся лениво —
    только при первом реальном вызове, поэтому сам по себе конструктор дешёвый.

    Поток: дешёвые сенсоры (текст промпта, OCR + текст OCR, визуальные
    классификаторы) собирают сигналы → PolicyJudge сводит их в одну
    откалиброванную оценку на категорию (и при необходимости эскалирует на
    ShieldGemma) → DecisionEngine выносит вердикт allow/review/block.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.text_guard = TextGuardHeuristic(
            enabled=self.settings.enable_text_guard,
            model_id=self.settings.text_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.ocr = OCRAdapter(
            enabled=self.settings.enable_ocr,
            tesseract_cmd=self.settings.tesseract_cmd,
            tessdata_dir=self.settings.tessdata_dir,
        )
        self.visual = VisualClassifierAdapter(
            enabled=self.settings.enable_visual_classifier,
            model_id=self.settings.visual_model_id,
            cache_dir=self.settings.hf_cache_dir,
            calibration_floor=self.settings.calibration_floor,
        )
        self.explicit = ExplicitContentAdapter(
            enabled=self.settings.enable_explicit_detector,
            model_id=self.settings.explicit_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.policy_judge = PolicyJudge(
            enabled=self.settings.enable_policy_judge,
            model_id=self.settings.policy_judge_model_id,
            review_threshold=self.settings.review_threshold,
            block_threshold=self.settings.block_threshold,
        )
        self.decision_engine = DecisionEngine(
            block_threshold=self.settings.block_threshold,
            review_threshold=self.settings.review_threshold,
        )

    def moderate(self, request: ModerationRequest) -> ModerationResponse:
        # Каждый сенсор возвращает SignalResult — единый «конверт» с категориями
        # и их оценками. Мы собираем их в список signals и в конце сводим воедино.
        image = load_image(request)
        signals = [self.text_guard.moderate(request.prompt)]

        if image is not None:
            # 1) OCR: вытаскиваем текст, вшитый в саму картинку.
            ocr_signal = self.ocr.extract(image)
            signals.append(ocr_signal)
            # 2) Найденный OCR-текст прогоняем через текстовый гард (реальный, не
            #    заглушка). Переименовываем сигнал, чтобы было видно, что это
            #    проверка именно OCR-текста, а не основного промпта.
            if ocr_signal.text:
                ocr_text = "\n".join(ocr_signal.text)
                ocr_text_result = self.text_guard.moderate(ocr_text)
                ocr_text_result.name = "ocr_text_guard_heuristic"
                signals.append(ocr_text_result)

            # 3) Два визуальных сенсора: откалиброванный zero-shot классификатор и
            #    узкоспециализированный детектор NSFW.
            signals.append(self.visual.moderate(image))
            signals.append(self.explicit.moderate(image))

        # 4) Судья: сводит все собранные сигналы в policy_fusion (+ эскалация на
        #    ShieldGemma при необходимости). Работает и для текст-онли запросов.
        signals.extend(
            self.policy_judge.judge(
                image=image,
                prompt=request.prompt,
                signals=signals,
                stage=request.stage,
            )
        )

        # 5) Финальное решение allow / review / block по порогам.
        return self.decision_engine.decide(request, signals)
