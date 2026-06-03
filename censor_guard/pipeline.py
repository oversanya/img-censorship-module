from __future__ import annotations

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.ocr import OCRAdapter
from censor_guard.adapters.policy_judge import PolicyJudgeFacade
from censor_guard.adapters.text_stub import TextGuardStub
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
    Метод moderate() прогоняет один запрос через все сенсоры и сводит их
    сигналы в итоговый вердикт через DecisionEngine.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.text_guard = TextGuardStub()
        self.ocr = OCRAdapter(
            enabled=self.settings.enable_ocr,
            tesseract_cmd=self.settings.tesseract_cmd,
            tessdata_dir=self.settings.tessdata_dir,
        )
        self.visual = VisualClassifierAdapter(
            enabled=self.settings.enable_visual_classifier,
            model_id=self.settings.visual_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.explicit = ExplicitContentAdapter(
            enabled=self.settings.enable_explicit_detector,
            model_id=self.settings.explicit_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.policy_judge = PolicyJudgeFacade(
            enabled=self.settings.enable_policy_judge,
            model_id=self.settings.policy_judge_model_id,
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
            # 2) Если на картинке нашёлся текст — прогоняем его через текстовый
            #    гард (сейчас это заглушка, см. text_stub.py). Переименовываем
            #    сигнал, чтобы в ответе было видно, что это проверка OCR-текста,
            #    а не основного промпта.
            if ocr_signal.text:
                ocr_text = "\n".join(ocr_signal.text)
                ocr_text_result = self.text_guard.moderate(ocr_text)
                ocr_text_result.name = "ocr_text_guard_stub"
                signals.append(ocr_text_result)

            # 3) Два визуальных сенсора: zero-shot мульти-классификатор по нашей
            #    таксономии и узкоспециализированный детектор NSFW.
            signals.append(self.visual.moderate(image))
            signals.append(self.explicit.moderate(image))

            # 4) Policy judge — «арбитр», который слитно переоценивает все
            #    собранные сигналы (эвристика) или, в будущем, спросит мультимодальную
            #    модель ShieldGemma. Получает signals целиком, поэтому идёт последним.
            policy_signal = self.policy_judge.moderate(
                image=image,
                prompt=request.prompt,
                signals=signals,
            )
            signals.append(policy_signal)

        # 5) Финальное решение allow / review / block по порогам.
        return self.decision_engine.decide(request, signals)
