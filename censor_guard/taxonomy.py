from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategorySpec:
    code: str
    label: str
    description: str
    hard_block: bool


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        code="sexual",
        label="sexual or explicit content",
        description="nudity, pornography, or sexualized imagery",
        hard_block=True,
    ),
    CategorySpec(
        code="violence_gore",
        label="violence or gore",
        description="physical violence, gore, blood, or mutilation",
        hard_block=True,
    ),
    CategorySpec(
        code="self_harm",
        label="self-harm or suicide content",
        description="self-harm, suicide, or injurious behavior toward self",
        hard_block=True,
    ),
    CategorySpec(
        code="hate_extremism",
        label="hate or extremism",
        description="hate symbols, extremist propaganda, slurs, or hateful imagery",
        hard_block=True,
    ),
    CategorySpec(
        code="illegal_activity",
        label="illegal activity",
        description="crime enablement, weapon misuse, drugs, or criminal acts",
        hard_block=True,
    ),
    CategorySpec(
        code="harassment",
        label="harassment or abuse",
        description="targeted insults, humiliation, or threatening abuse",
        hard_block=False,
    ),
    CategorySpec(
        code="deception_fraud",
        label="deception or fraud",
        description="impersonation, forged claims, scams, or misleading offers",
        hard_block=False,
    ),
    CategorySpec(
        code="political_persuasion",
        label="political persuasion",
        description="political campaigning, persuasion, or election influence",
        hard_block=False,
    ),
    CategorySpec(
        code="health_misinformation",
        label="health misinformation",
        description="misleading medical claims, unsafe advice, or false health guidance",
        hard_block=False,
    ),
    CategorySpec(
        code="spam_scams",
        label="spam or scam promotion",
        description="spam creatives, phishing, or suspicious marketing content",
        hard_block=False,
    ),
    CategorySpec(
        code="shocking",
        label="shocking disturbing content",
        description="graphic, revolting, or disturbing imagery",
        hard_block=True,
    ),
)


CATEGORY_BY_CODE = {spec.code: spec for spec in CATEGORY_SPECS}
VISUAL_LABEL_TO_CODE = {spec.label: spec.code for spec in CATEGORY_SPECS}

# Несколько нейтральных «якорей безопасности». Zero-shot классификатор softmax-ит
# оценки по ВСЕМ переданным меткам, поэтому без сильного безопасного якоря у любой
# картинки всегда «побеждает» какая-то категория нарушения. Чем больше формулировок
# безопасного, тем устойчивее базовая линия, относительно которой мы калибруем
# (см. calibration.py). Первый элемент сохранён как SAFE_VISUAL_LABEL для совместимости.
SAFE_VISUAL_LABELS = (
    "safe everyday image",
    "neutral ordinary photo",
    "harmless normal content",
    "wholesome safe-for-work image",
)
SAFE_VISUAL_LABEL = SAFE_VISUAL_LABELS[0]
SAFE_VISUAL_LABEL_SET = set(SAFE_VISUAL_LABELS)
VISUAL_LABELS = [spec.label for spec in CATEGORY_SPECS] + list(SAFE_VISUAL_LABELS)
HARD_BLOCK_CATEGORIES = {spec.code for spec in CATEGORY_SPECS if spec.hard_block}
SOFT_REVIEW_CATEGORIES = {spec.code for spec in CATEGORY_SPECS if not spec.hard_block}


# Текстовые лексиконы для базового (без ML) текстового классификатора.
# Хранятся как СТЕМЫ в нижнем регистре: совпадение проверяется по границе слова с
# учётом, что у русского богатая морфология (поэтому "убива" поймает "убивать",
# "убивают" и т.д.). Это осознанно простой, объяснимый baseline — он не заменяет
# ML-модель, но даёт реальную (а не заглушечную) фильтрацию текста промпта и OCR.
# Ключ — код категории, значение — кортеж стемов (ru + en вперемешку).
TEXT_LEXICONS: dict[str, tuple[str, ...]] = {
    "sexual": (
        "порн", "секс", "эроти", "обнаж", "голая", "голый", "голую", "интим",
        "шлюх", "проститут", "бордель", "минет", "оргия", "разврат", "мастурб",
        "совокупл", "сексуальн", "genital", "porn", "nude", "naked", "explicit sex",
        "nsfw", "hentai", "blowjob", "anal", "вагин", "пенис", "сосок", "сиськ",
        "slut", "whore", "prostitute", "escort", "masturbat", "orgy",
    ),
    "violence_gore": (
        "убива", "убий", "расчлен", "кровь", "избива", "пытк", "резать",
        "застрел", "зарез", "обезглав", "kill", "murder", "gore", "behead",
        "torture", "bloodbath", "stab", "massacre", "mutilat",
    ),
    "self_harm": (
        "суицид", "самоубий", "вскры вены", "вскрыть вены", "порезать себя",
        "повесит", "self-harm", "self harm", "suicide", "kill myself",
        "cut myself", "end my life", "hang myself",
    ),
    "hate_extremism": (
        "террор", "экстремизм", "нацис", "фашис", "джихад", "теракт",
        "взорвать", "geno", "terror", "extremis", "nazi", "jihad",
        "ethnic cleansing", "white power", "лозунг ненавист",
    ),
    "illegal_activity": (
        "наркот", "героин", "кокаин", "метамфетамин", "взрывчат", "оружие купить",
        "купить ствол", "поддель", "взлом", "украсть", "drugs to buy", "cocaine",
        "heroin", "meth recipe", "make a bomb", "build a bomb", "buy a gun illegally",
        "counterfeit", "hack into",
    ),
    "harassment": (
        "ничтожеств", "тупая тварь", "сдохни", "уебищ", "оскорбл", "унижа",
        "loser", "kill yourself", "you are worthless", "pathetic idiot",
        "harass", "bully",
    ),
    "deception_fraud": (
        "развод на деньг", "мошенн", "обман", "фишинг", "поддельный счёт",
        "phishing", "scam", "fraud", "fake invoice", "wire me money",
        "impersonat",
    ),
    "spam_scams": (
        "выиграл приз", "беспроигрыш", "быстрый заработок", "казино бонус",
        "перейди по ссылк", "free money", "you won", "click this link",
        "limited offer", "get rich quick", "crypto giveaway",
    ),
}


def is_known_category(code: str) -> bool:
    return code in CATEGORY_BY_CODE

