from __future__ import annotations

from censor_guard.fusion import FUSION_SIGNAL_NAME, fuse
from censor_guard.schemas import SignalResult


class ShieldGemmaJudge:
    """Точка интеграции мультимодального судьи (ShieldGemma) как ЭСКАЛАЦИОННОГО слоя.

    Это НЕ аналог CLIP и не «ещё один классификатор по сходству»: ShieldGemma —
    policy-aware VLM, которому отдают картинку + собранные улики и просят вынести
    обоснованное решение по политике. Поэтому он вызывается не на каждый запрос, а
    только на пограничных случаях (см. PolicyJudge.escalation_needed), где дешёвые
    сенсоры не уверены, или на стадии output, где цена ошибки выше.

    Сейчас инференс не подключён → всегда status="skipped" (честно, без подмены
    результата). Когда модель подключат, метод вернёт status="ok" с оценками по
    категориям, и fusion учтёт их как сильный сигнал (вес 1.0).
    """

    name = "policy_judge_shieldgemma"

    def __init__(self, enabled: bool, model_id: str) -> None:
        self.enabled = enabled
        self.model_id = model_id

    def moderate(self, image, prompt: str | None, evidence: dict[str, float]) -> SignalResult:
        if not self.enabled:
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="ShieldGemma escalation disabled by configuration.",
            )
        return SignalResult(
            name=self.name,
            status="skipped",
            reason=(
                "ShieldGemma escalation point is enabled but model inference is not wired in this build. "
                "Verdict relies on calibrated sensor fusion until the model is attached."
            ),
            raw={"model_id": self.model_id, "prompt_present": bool(prompt), "evidence": evidence},
        )


class PolicyJudge:
    """Арбитр пайплайна. Делает две вещи:

    1) Сводит все сигналы сенсоров в одну откалиброванную оценку на категорию
       (взвешенный noisy-OR, censor_guard.fusion) — это заменяет прежнюю эвристику
       с магическими числами и бессмысленным evidence_count.
    2) На пограничных случаях эскалирует на ShieldGemma и, если тот ответил,
       пере-сводит fusion уже с его участием.

    Возвращает список SignalResult: сначала (опционально) сигнал ShieldGemma, затем
    итоговый сигнал `policy_fusion`, который потребляет DecisionEngine.
    """

    def __init__(
        self,
        enabled: bool,
        model_id: str,
        review_threshold: float,
        block_threshold: float,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.shieldgemma = ShieldGemmaJudge(enabled=enabled, model_id=model_id)
        self.review_threshold = review_threshold
        self.block_threshold = block_threshold
        self.weights = weights

    def escalation_needed(self, fused: dict[str, float], stage: str) -> bool:
        # «Серая зона»: модель не уверена — review-уровень есть, но до уверенного
        # блока не дотянули. Именно здесь мнение policy-aware судьи ценнее всего.
        gray_zone = any(
            self.review_threshold <= score < self.block_threshold for score in fused.values()
        )
        # На выходе генерации цена пропуска выше — эскалируем при любом ненулевом риске.
        output_risk = stage == "output" and any(
            score >= self.review_threshold for score in fused.values()
        )
        return gray_zone or output_risk

    def _build_fusion_signal(self, fusion, escalation: dict) -> SignalResult:
        categories = fusion.scores()
        contributions = {code: cat.contributions for code, cat in fusion.categories.items()}
        agreement = {code: cat.agreement for code, cat in fusion.categories.items()}
        return SignalResult(
            name=FUSION_SIGNAL_NAME,
            status="ok",
            categories=categories,
            reason="Fused calibrated sensor evidence via weighted noisy-OR.",
            raw={
                "mode": "weighted_noisy_or",
                "contributions": contributions,
                "agreement": agreement,
                "escalation": escalation,
            },
        )

    def judge(
        self,
        image,
        prompt: str | None,
        signals: list[SignalResult],
        stage: str,
    ) -> list[SignalResult]:
        fusion = fuse(signals, weights=self.weights)
        fused_scores = fusion.scores()

        out: list[SignalResult] = []
        escalation = {"attempted": False, "needed": self.escalation_needed(fused_scores, stage)}

        if escalation["needed"]:
            escalation["attempted"] = True
            sg_signal = self.shieldgemma.moderate(image=image, prompt=prompt, evidence=fused_scores)
            escalation["shieldgemma_status"] = sg_signal.status
            out.append(sg_signal)
            if sg_signal.status == "ok":
                # Судья высказался — пере-сводим fusion уже с его участием.
                fusion = fuse(signals + [sg_signal], weights=self.weights)

        out.append(self._build_fusion_signal(fusion, escalation))
        return out
