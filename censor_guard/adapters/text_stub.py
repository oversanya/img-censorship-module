from __future__ import annotations

import re
from dataclasses import dataclass

from censor_guard.schemas import SignalResult


@dataclass(frozen=True)
class TextRule:
    category: str
    score: float
    patterns: tuple[str, ...]


TEXT_RULES: tuple[TextRule, ...] = (
    TextRule(
        "sexual_minors",
        0.98,
        (
            r"\b(?:child|minor|teen)\s+(?:porn|nude|sex)",
            r"детск\w*\s+порно",
            r"(?:несовершеннолетн\w*|подрост\w*|школьн\w*|детск\w*).*(?:секс|порно|нюд|обнажен|сексуализ)",
            r"(?:секс|порно|нюд|обнажен|сексуализ).*(?:несовершеннолетн\w*|подрост\w*|школьн\w*|детск\w*)",
        ),
    ),
    TextRule(
        "sexual",
        0.93,
        (
            r"\b(?:porn|porno|nude|nsfw|explicit|sex scene)\b",
            r"\b(?:эротик\w*|порно\w*|обнажен\w*|интим\w*|нюд\w*|18\+|adult)\b",
            r"\b(?:бляд\w*|блядот\w*)\b",
        ),
    ),
    TextRule(
        "hate_extremism",
        0.96,
        (
            r"\b(?:terrorist|terrorism|isis|nazi|extremist)\b",
            r"\b(?:террорист|терроризм|экстремист|нацист|свастик|радикальн)\w*\b",
            r"\b(?:запрещенн\w*|экстремистск\w*|нацистск\w*)\s+символ\w*\b",
            r"\b(?:террористическ\w*|экстремистск\w*)\s+(?:организац\w*|пропаганд\w*|баннер\w*)\b",
        ),
    ),
    TextRule(
        "violence_gore",
        0.94,
        (
            r"\b(?:gore|blood|torture|mutilation|execution)\b",
            r"\b(?:кровь|кров\w*|пытк\w*|пыток|казн\w*|расчлен\w*|жесток\w*|убийств\w*|избиен\w*)\b",
        ),
    ),
    TextRule(
        "self_harm",
        0.92,
        (
            r"\b(?:suicide|self[- ]?harm|hang myself)\b",
            r"\b(?:суицид\w*|самоповрежд\w*|порез\w*|романтизац\w*\s+суицид\w*|покончить с собой)\b",
        ),
    ),
    TextRule(
        "drugs",
        0.91,
        (
            r"\b(?:drugs|narcotics|cocaine|heroin|meth)\b",
            r"\b(?:наркотик\w*|наркот\w*|наркот[ауы]|нарик\w*|нар[ыи]г\w*|кокаин\w*|героин\w*|метадон\w*|закладк\w*)\b",
        ),
    ),
    TextRule(
        "weapons_crime",
        0.91,
        (
            r"\b(?:weapon|explosive|bomb|molotov|criminal instruction)\b",
            r"\b(?:оружи\w*|взрывчатк\w*|бомб\w*|криминальн\w*\s+инструкц|преступлен\w*|преступн\w*)\b",
        ),
    ),
    TextRule(
        "deception_fraud",
        0.95,
        (
            r"\b(?:phishing|scam|fraud|social engineering|fake login)\b",
            r"\b(?:фишинг\w*|фишингов\w*|скам\w*|мошеннич\w*|социальн\w*\s+инженер\w*|обман\w*|краж\w*\s+логин\w*)\b",
        ),
    ),
    TextRule(
        "forged_documents",
        0.94,
        (
            r"\b(?:fake|forged)\s+(?:passport|certificate|contract|card|statement)\b",
            r"\b(?:поддельн\w*|фальшив\w*)\s+(?:паспорт|справк\w*|договор\w*|карт\w*|выписк\w*)\b",
            r"\b(?:банковск\w*\s+справк\w*|фальшив\w*\s+договор\w*)\b",
        ),
    ),
    TextRule(
        "personal_biometric_data",
        0.88,
        (
            r"\b(?:passport data|card number|cvv|biometric|face id|fingerprint)\b",
            r"\b(?:персональн\w*\s+данн\w*|паспортн\w*\s+данн\w*|номер\s+карт\w*|cvv|биометр\w*|отпечатк\w*)\b",
            r"\b(?:паспорт\w*|лиц\w*)\s+с\s+(?:персональн\w*\s+данн\w*|биометр\w*)\b",
        ),
    ),
    TextRule(
        "fraudulent_qr_payment",
        0.95,
        (
            r"\b(?:fake qr|payment qr|fake push|push notification)\b",
            r"\b(?:фейков\w*|поддельн\w*|мошенническ\w*)\s+(?:qr|куар|push|пуш|уведомлен\w*|платежн\w*\s+форм\w*)\b",
            r"\b(?:qr|куар)\s*(?:код\w*)?\s+(?:для\s+)?(?:оплат\w*|перевод\w*|платеж\w*)\b",
            r"\b(?:платежн\w*\s+форм\w*|push\s*уведомлен\w*|пуш\s*уведомлен\w*)\b",
        ),
    ),
    TextRule(
        "financial_misleading",
        0.72,
        (
            r"\b(?:guaranteed loan|guaranteed cashback|zero risk|instant profit)\b",
            r"\b(?:гарантированн\w*\s+(?:доход|прибыл|кредит|кэшбек|одобрен\w*)|без\s+риска|нулев\w*\s+риск|мгновенн\w*\s+прибыл)\b",
        ),
    ),
    TextRule(
        "investment_manipulation",
        0.76,
        (
            r"\b(?:pump and dump|market manipulation|guaranteed investment)\b",
            r"\b(?:памп|дамп|манипуляц\w*\s+рынк\w*|гарантированн\w*\s+инвестиц|гарантированн\w*\s+прибыл\w*)\b",
        ),
    ),
    TextRule(
        "political_persuasion",
        0.68,
        (
            r"\b(?:vote for|election campaign|political ad)\b",
            r"\b(?:голосуй\s+за|выборн\w*\s+агитац\w*|политическ\w*\s+агитац\w*|призыв\w*\s+поддержать\s+парт\w*|политическ\w*\s+символик\w*)\b",
        ),
    ),
    TextRule(
        "brand_ip_abuse",
        0.66,
        (
            r"\b(?:use beta bank logo|copy logo|brand impersonation|trademark)\b",
            r"\b(?:логотип\s+бета|скопируй\s+логотип|чуж\w*\s+бренд|товарн\w*\s+знак|незаконн\w*\s+использован\w*)\b",
        ),
    ),
    TextRule(
        "official_interface_impersonation",
        0.78,
        (
            r"\b(?:fake government|fake bank interface|official notice)\b",
            r"\b(?:фейков\w*|поддельн\w*)\s+(?:госуслуг\w*|портал\w*|интерфейс\s+банк\w*|официальн\w*\s+уведомлен|окно\s+регулятор\w*)\b",
            r"\b(?:госуслуг\w*|регулятор\w*|официальн\w*\s+экран\w*)\b",
        ),
    ),
    TextRule(
        "bank_reputation_risk",
        0.64,
        (
            r"\b(?:bank steals|defame bank|defame employee)\b",
            r"\b(?:банк\s+вору\w*|порочащ\w*\s+банк|клевет\w*\s+на\s+сотрудник|унижен\w*\s+клиент\w*|унижени\w*\s+клиент\w*)\b",
        ),
    ),
    TextRule(
        "gambling_fast_money",
        0.68,
        (
            r"\b(?:casino|betting|sportsbook|get rich quick)\b",
            r"\b(?:казино|ставк\w*|букмекер\w*|букмекерск\w*|быстр\w*\s+деньг\w*|легк\w*\s+выигрыш\w*|быстр\w*\s+заработ\w*)\b",
        ),
    ),
    TextRule(
        "sanctions_geopolitical",
        0.64,
        (
            r"\b(?:sanctioned entity|military propaganda|geopolitical)\b",
            r"\b(?:санкционн\w*|военн\w*\s+пропаганд\w*|геополитическ\w*\s+(?:риск|конфликт|символик)\w*)\b",
        ),
    ),
    TextRule(
        "shocking",
        0.90,
        (
            r"\b(?:shocking|disturbing|revolting|body horror)\b",
            r"\b(?:шок\s*контент|шоков\w*|отвратительн\w*|пугающ\w*|телесн\w*\s+ужас|револтинг\w*)\b",
        ),
    ),
    TextRule(
        "harassment",
        0.91,
        (
            r"\b(?:whore|slut|bitch|moron|idiot|scum)\b",
            r"\b(?:shlyuh\w*|shluh\w*|suka|sucka)\b",
            r"\b(?:шлюх\w*|су[кч]\w*|мраз\w*|твар\w*|долбо[её]б\w*|у[её]б\w*|гандон\w*|хуесос\w*)\b",
        ),
    ),
    TextRule(
        "discrimination_hate",
        0.93,
        (
            r"\b(?:hate speech|racial slur|dehumanize|homophobic slur|pidor\w*|pedor\w*)\b",
            r"\b(?:дискриминац\w*|hate speech|разжиган\w*\s+ненавист\w*)\b",
            r"\b(?:пид[оа]р\w*|пидорас\w*|педик\w*)\b",
        ),
    ),
)


LOOKALIKE_CHARS = str.maketrans(
    {
        "ё": "е",
        "0": "о",
        "3": "з",
        "4": "ч",
        "6": "б",
        "@": "а",
        "$": "с",
        "a": "а",
        "c": "с",
        "e": "е",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)

LEET_CHARS = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "!": "i",
    }
)


def normalize_text_variants(text: str) -> tuple[str, ...]:
    lowered = text.casefold().replace("ё", "е")
    leet = lowered.translate(LEET_CHARS)
    lookalike = lowered.translate(LOOKALIKE_CHARS)
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    collapsed_leet = re.sub(r"\s+", " ", leet).strip()
    collapsed_lookalike = re.sub(r"\s+", " ", lookalike).strip()
    compact = re.sub(r"[^0-9a-zа-я]+", "", lowered, flags=re.UNICODE)
    compact_leet = re.sub(r"[^0-9a-zа-я]+", "", leet, flags=re.UNICODE)
    compact_lookalike = re.sub(r"[^0-9a-zа-я]+", "", lookalike, flags=re.UNICODE)
    variants = (
        lowered,
        collapsed,
        leet,
        collapsed_leet,
        lookalike,
        collapsed_lookalike,
        compact,
        compact_leet,
        compact_lookalike,
    )
    return tuple(dict.fromkeys(variant for variant in variants if variant))


class TextGuardStub:
    """Легковесный эвристический текстовый гард для демо-контура.

    Он не заменяет ML/LLM-судью, но закрывает базовый хакатонный сценарий:
    опасный промпт или OCR-текст сразу дают категорию, оценку и объяснимый след.
    """

    name = "text_guard_heuristic"

    def moderate(self, text: str | None) -> SignalResult:
        if not text:
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="No prompt text supplied.",
            )
        text_variants = normalize_text_variants(text)
        categories: dict[str, float] = {}
        matched_rules: dict[str, list[str]] = {}
        for rule in TEXT_RULES:
            matches = [
                pattern
                for pattern in rule.patterns
                if any(re.search(pattern, variant, flags=re.IGNORECASE) for variant in text_variants)
            ]
            if not matches:
                continue
            categories[rule.category] = max(categories.get(rule.category, 0.0), rule.score)
            matched_rules.setdefault(rule.category, []).extend(matches)

        if categories:
            reason = "Heuristic text guard matched policy keywords in prompt or OCR text."
        else:
            reason = "Heuristic text guard found no policy keyword matches."
        return SignalResult(
            name=self.name,
            status="ok",
            categories=categories,
            reason=reason,
            raw={"mode": "keyword_heuristic", "text_length": len(text), "matched_rules": matched_rules},
        )
