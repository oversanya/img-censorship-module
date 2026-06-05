from __future__ import annotations

import os
import re
from pathlib import Path

from censor_guard.schemas import SignalResult
from censor_guard.taxonomy import LLAVAGUARD_CATEGORY_MAP


# Промпт-таксономия, на которой обучали LlavaGuard v1.2. Текст важно держать близко
# к оригиналу: модель выучила именно эти формулировки и формат JSON-ответа. Vision-
# токены доминируют над текстом, так что длина политики на латентность почти не
# влияет — отдаём полную, ради качества.
LLAVAGUARD_POLICY = """Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment:
Should not: Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics or that encourages discrimination or harassment against them.
Can: Contain educational content about discrimination, humiliation, and harassment, or discuss historical events involving hate and harassment.
O2: Violence, Harm, or Cruelty:
Should not: Contain content of people planning or engaging in violence, or that encourages people to engage in violence or cruelty, including harm to others.
Can: Contain educational content about violence, harm, and cruelty, or discuss historical events involving violence.
O3: Sexual Content:
Should not: Contain sexually explicit (i.e., erotic) content.
Can: Contain content related to sexuality and sexual education, or general information about sexual activities, or content of people in a loving and affectionate manner that is not sexually explicit.
O4: Nudity Content:
Should not: Contain visible genitalia or explicit nudity content.
Can: Contain artistic nudity without visible genitalia, or educational and general information on the human body.
O5: Criminal Planning:
Should not: Contain content of people planning or engaging in criminal activities, or that encourages people to engage in criminal activities or condones criminal behavior.
Can: Help people understand how crimes are committed (without a roadmap), or educate on crime prevention.
O6: Weapons or Substance Abuse:
Should not: Contain illegal weapons with intent to promote violence, trafficking or use of illegal drugs, or the creation of illegal substances.
Can: Contain weapons in a legal and safe context, or factual educational information about regulated substances without promoting their use.
O7: Self-Harm:
Should not: Contain content that could encourage people to harm themselves, or that promotes or condones self-harm, including suicide, cutting, and eating disorders.
Can: Contain educational content about self-harm and suicide, or resources guiding individuals toward professional help.
O8: Animal Cruelty:
Should not: Contain content that could encourage people to harm animals, that promotes animal cruelty, or that contains explicit animal cruelty.
Can: Contain general educational information regarding animal cruelty.
O9: Disasters or Emergencies:
Should not: Contain content about deceased or harmed individuals as a result of disasters, or content about suffering, cruelty, or exploitation during such events.
Can: Contain respectful news coverage and educational content about disasters or emergencies.
Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the content and justify why it is safe or unsafe with reference to the policy category.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}"""

_RATING_RE = re.compile(r'"rating"\s*:\s*"([^"]+)"', re.IGNORECASE)
_CATEGORY_RE = re.compile(r'"category"\s*:\s*"\s*(O[1-9])', re.IGNORECASE)


def interpret_llavaguard_output(text: str, unsafe_score: float) -> tuple[dict[str, float], dict]:
    """Чистая (без модели) интерпретация текстового ответа LlavaGuard → оценки.

    LlavaGuard отдаёт ОДНУ доминирующую категорию + бинарный rating, поэтому на
    выходе ноль или одна категория нашей таксономии. `rating`/`category` стоят в
    JSON первыми, так что вытаскиваем их регуляркой — это устойчиво даже если вывод
    обрезан по max_new_tokens и `rationale` не закрыт. Возвращаем (categories, info)."""
    rating_m = _RATING_RE.search(text)
    cat_m = _CATEGORY_RE.search(text)
    rating = (rating_m.group(1).strip().lower() if rating_m else "")
    ocode = (cat_m.group(1).upper() if cat_m else None)
    info = {"rating": rating or None, "llavaguard_category": ocode, "text": text.strip()}

    if rating != "unsafe":
        return {}, info
    code = LLAVAGUARD_CATEGORY_MAP.get(ocode or "")
    if not code:
        # Unsafe, но категория NA/нечитабельна — фиксируем факт, но не плодим оценку.
        return {}, info
    info["mapped_code"] = code
    return {code: float(unsafe_score)}, info


class LlavaGuardAdapter:
    """Всегда работающий визуальный safety-судья (LlavaGuard-0.5B-OV по умолчанию).

    В отличие от zero-shot CLIP — это ОБУЧЕННЫЙ policy-aware VLM: ему отдают
    картинку + таксономию политики, он возвращает структурированный JSON
    {rating, category, rationale}. Мы переводим вердикт в оценку по нашей
    таксономии (taxonomy.LLAVAGUARD_CATEGORY_MAP) и отдаём как обычный сигнал.

    Это второй независимый визуальный сенсор рядом с CLIP: он закрывает дыру, где
    у не-sexual визуальных категорий был ровно один (слабый) голос, и позволяет
    набрать осмысленное согласие сенсоров. Надёжность отражена весом в fusion.

    Стоимость: на CPU это ~10–30 с/картинку (доминирует префилл vision-токенов).
    Ленивая загрузка + кэш ошибки: при недоступности transformers/torchvision/весов
    тихо деградируем (skipped/error), не роняя сервис.
    """

    name = "llava_guard"

    def __init__(
        self,
        enabled: bool,
        model_id: str,
        cache_dir: str,
        max_new_tokens: int = 24,
        unsafe_score: float = 0.9,
    ) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.unsafe_score = unsafe_score
        self._model = None
        self._processor = None
        self._prompt = None  # кэш применённого chat-template
        self._load_error: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        except ImportError:
            return None
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_dir)
            import torch

            self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_id, dtype=torch.float32, cache_dir=self.cache_dir
            )
            self._model.eval()
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": LLAVAGUARD_POLICY}]}
            ]
            self._prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = str(exc)
            self._model = None
            return None
        return self._model

    def moderate(self, image) -> SignalResult:
        if not self.enabled:
            return SignalResult(name=self.name, status="skipped", reason="LlavaGuard disabled by configuration.")
        model = self._load()
        if model is None:
            if self._load_error:
                return SignalResult(name=self.name, status="error", reason=f"LlavaGuard load failed: {self._load_error}")
            return SignalResult(name=self.name, status="skipped", reason="transformers/torchvision is not installed.")

        try:
            import torch

            inputs = self._processor(text=self._prompt, images=image, return_tensors="pt")
            n_in = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            text = self._processor.decode(out[0][n_in:], skip_special_tokens=True)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return SignalResult(name=self.name, status="error", reason=f"LlavaGuard inference failed: {exc}")

        categories, info = interpret_llavaguard_output(text, self.unsafe_score)
        reason = (
            f"LlavaGuard: {info.get('rating') or 'no-rating'}"
            + (f" → {info['mapped_code']}" if "mapped_code" in info else "")
        )
        return SignalResult(name=self.name, status="ok", categories=categories, reason=reason, raw=info)
