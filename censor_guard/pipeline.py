from __future__ import annotations

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.llava_guard import LlavaGuardAdapter
from censor_guard.adapters.ocr import OCRAdapter
from censor_guard.adapters.policy_judge import PolicyJudge
from censor_guard.adapters.robust_guard import RobustGuardAdapter
from censor_guard.adapters.text_classifier import TextGuard
from censor_guard.adapters.visual_classifier import VisualClassifierAdapter
from censor_guard.config import Settings
from censor_guard.decision import DecisionEngine
from censor_guard.image_utils import load_image
from censor_guard.schemas import ModerationRequest, ModerationResponse
from censor_guard.guardrails.image_guard import ImageAnalyzer
from censor_guard.guardrails.string_guard import StringGuard


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
        self.text_guard = TextGuard(
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
        self.llava_guard = LlavaGuardAdapter(
            enabled=self.settings.enable_llava_guard,
            model_id=self.settings.llava_guard_model_id,
            cache_dir=self.settings.hf_cache_dir,
            max_new_tokens=self.settings.llava_guard_max_new_tokens,
            unsafe_score=self.settings.llava_guard_unsafe_score,
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

        self.string_guard = StringGuard(enabled=self.settings.enable_injection_revealer)
        self.image_analyzer = ImageAnalyzer(enabled=self.settings.enable_image_sanitizer)
        
        self.robust_guard = RobustGuardAdapter(
            enabled=self.settings.enable_robust_guard,
            probe_path=self.settings.robust_probe_path,
            model_dir=self.settings.robust_model_dir,
        )
        self.robust_unsafe_min = self.settings.robust_unsafe_min
        self.review_threshold = self.settings.review_threshold
        self.use_guardrails = True

    def assess(self, image, prompt, stage):

        signals = [self.text_guard.moderate(prompt)]

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
                ocr_text_result.name = "ocr_text_guard"
                signals.append(ocr_text_result)

            # 3) Визуальные сенсоры: откалиброванный zero-shot классификатор (CLIP),
            #    узкоспециализированный детектор NSFW и обученный policy-aware судья
            #    LlavaGuard (второй независимый голос по всем визуальным категориям).
            signals.append(self.visual.moderate(image))
            signals.append(self.explicit.moderate(image))
            signals.append(self.llava_guard.moderate(image))

        # 4) Судья: сводит все собранные сигналы в policy_fusion (+ эскалация на
        #    ShieldGemma при необходимости). Работает и для текст-онли запросов.
        signals.extend(
            self.policy_judge.judge(
                image=image,
                prompt=prompt,
                signals=signals,
                stage=stage,
            )
        )

        return signals

    def get_decision(self, request: ModerationRequest, image=None, prompt=None) -> ModerationResponse:
        # Каждый сенсор возвращает SignalResult — единый «конверт» с категориями
        # и их оценками. Мы собираем их в список signals и в конце сводим воедино.
        if image is None:
            image = load_image(request)
        if prompt is None:
            prompt = request.prompt

        signals = self.assess(image, prompt, request.stage)

        return self.decision_engine.decide(request, signals)

    def compare_guarded(self, request: ModerationRequest):

        resp_before = self.get_decision(request)
        conf_before = resp_before.confidence
        prompt = request.prompt

        image = load_image(request)
        sanitized = self.image_analyzer.process(image)["image"]

        resp_after = self.get_decision(request, image=sanitized)
        conf_after = resp_after.confidence

        if resp_after.verdict != resp_before.verdict and abs(conf_before - conf_after) > 0.4:
            return {
                "verdict": "block",
                "reason": "suspicious image detected, highly unstable\n",
                "conf_before": conf_before,
                "conf_after": conf_after,
                "decision_after": resp_after
            }

        # Вердикты совпали (картинка стабильна к санации) либо разошлись, но в
        # пределах допустимого разброса уверенности — несогласованности нет.
        return {
            "verdict": "pass",
            "reason": "no inconsistencies revealed",
            "conf_before": conf_before,
            "conf_after": conf_after,
            "decision_after": resp_after
        }

    def check_adversarial(self, image, main_unsafe_score: float):
        """Сверяет основной пайплайн с робастной моделью как детектором adversarial.

        `main_unsafe_score` — unsafe-оценка основного (не робастного) пайплайна на
        ИСХОДНОЙ картинке (confidence из get_decision). Возмущения, заточенные под
        основную модель, робастную почти не двигают. Поэтому опасен НАПРАВЛЕННЫЙ
        случай: робастная модель уверенно видит unsafe (P ≥ robust_unsafe_min), а
        основной пайплайн картинку пропускает (main_unsafe < review_threshold) —
        классический обход. Тогда блокируем как unsafe. Обратное расхождение
        (основная паникует, робастная спокойна) и так покрыто fail-safe-блоком и
        здесь не штрафуется (иначе много ложных блокировок на чистых картинках).

        Если робастная модель недоступна (нет robustbench/весов), индикатор молчит:
        verdict="pass", блок не навязываем."""
        robust_signal = self.robust_guard.moderate(image)
        if robust_signal.status != "ok":
            return {"verdict": "pass", "reason": "robust guard unavailable", "signal": robust_signal}

        robust_unsafe = float(robust_signal.raw["robust_unsafe"])
        main_unsafe = float(main_unsafe_score)
        divergence = robust_unsafe - main_unsafe  # знаковое: >0 = робастная «опаснее»
        is_adversarial = robust_unsafe >= self.robust_unsafe_min and main_unsafe < self.review_threshold

        result = {
            "robust_unsafe": robust_unsafe,
            "main_unsafe": main_unsafe,
            "divergence": divergence,
            "signal": robust_signal,
        }
        if is_adversarial:
            result["verdict"] = "block"
            result["reason"] = (
                "adversarial example suspected: robust model flags unsafe while main "
                f"pipeline passes (robust P(unsafe)={robust_unsafe:.3f} ≥ {self.robust_unsafe_min}, "
                f"main unsafe={main_unsafe:.3f} < {self.review_threshold})\n"
            )
        else:
            result["verdict"] = "pass"
            result["reason"] = "robust model does not contradict main pipeline"
        return result

    def check_text_injections(self, prompt):
        suspicious = self.string_guard.process(prompt)
        return suspicious
        

    def moderate(self, request: ModerationRequest) -> ModerationResponse:
        # Локально на запрос: output-стадия проверяет уже сгенерированный результат,
        # image-guardrail там не нужен. Не трогаем self, иначе один output-запрос
        # выключил бы guardrail для всех последующих (pipeline — общий синглтон).
        use_guardrails = self.use_guardrails and request.stage != "output"

        prompt = request.prompt
    
        if request.stage == "input":
            prompt_analysis = self.check_text_injections(prompt)
            if prompt_analysis['safe'] == False:
                return ModerationResponse(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    stage=request.stage,
                    verdict="block",
                    categories={},
                    confidence=list(prompt_analysis['confidence'].values())[0],
                    reason=prompt_analysis['reasons'],
                    notes=["input prompt guardrail"]
                )

        inconsistency_test = self.compare_guarded(request)

        if use_guardrails:
            # Робастная модель как детектор adversarial: сверяем её P(unsafe) с
            # unsafe-оценкой основного пайплайна на исходной картинке.
            image = load_image(request)
            if image is not None:
                adversarial_test = self.check_adversarial(image, inconsistency_test['conf_before'])
                if adversarial_test["verdict"] == "block":
                    return ModerationResponse(
                        request_id=request.request_id,
                        scenario=request.scenario,
                        stage=request.stage,
                        verdict="block",
                        categories=[],
                        confidence=round(adversarial_test['divergence'], 4),
                        reason=adversarial_test['reason'],
                        signals=[adversarial_test['signal']],
                        notes=["robust-model adversarial guardrail"],
                    )

            if inconsistency_test["verdict"] == "block":
                return ModerationResponse(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    stage=request.stage,
                    verdict="block",
                    categories={},
                    confidence=abs(inconsistency_test['conf_before'] - inconsistency_test['conf_after']),
                    reason=inconsistency_test['reason'],
                    notes=["input image guardrailing"]
                )

        return inconsistency_test['decision_after']
