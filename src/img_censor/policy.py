from pathlib import Path

import yaml


LLAVAGUARD_POLICY = """
Provide a safety assessment for the content provided by the user. Determine
compliance or violation against our safety policy by reviewing the following
policy categories:

O1: Hate, Humiliation, Harassment:
Should not contain hateful, derogatory, or demeaning content against protected
classes or content encouraging discrimination or harassment.

O2: Violence, Harm, or Cruelty:
Should not contain graphic violence, gore, cruelty, severe injury, or content
encouraging harm to people or animals.

O3: Sexual Content:
Should not contain sexually explicit or pornographic content.

O4: Nudity Content:
Should not contain explicit nudity or visible genitalia. Contextual artistic,
medical, or educational content should be treated as review when ambiguous.

O5: Criminal Planning:
Should not contain criminal planning, financial crime, fraud, theft,
kidnapping, or instructions that enable crimes.

O6: Weapons or Substance Abuse:
Should not promote illegal weapons, explosives, terrorism, illegal drugs,
substance abuse, or weapon use with intent to harm.

O7: Self-Harm:
Should not encourage, instruct, romanticize, or depict actionable self-harm or
suicide.

O8: Personal or Financial Data:
Should not reveal or generate personal data, bank card numbers, account
numbers, passports, IDs, addresses, phone numbers, or other sensitive financial
information.

O9: Bank Brand/Reputation:
Should not depict bank branding in a hateful, sexual, violent, fraudulent, or
humiliating context.

Assessment steps:
1. Select a safety rating: "Unsafe" if the content violates the policy,
   otherwise "Safe".
2. Assign one category. If none applies, use "NA: None applying".
3. Provide a short rationale grounded in the visible content and policy.

Return only valid JSON:
{
  "rating": "Unsafe" | "Safe",
  "category": "O1: Hate, Humiliation, Harassment" | "O2: Violence, Harm, or Cruelty" | "O3: Sexual Content" | "O4: Nudity Content" | "O5: Criminal Planning" | "O6: Weapons or Substance Abuse" | "O7: Self-Harm" | "O8: Personal or Financial Data" | "O9: Bank Brand/Reputation" | "NA: None applying",
  "rationale": "short explanation"
}
""".strip()


LLAVAGUARD_CATEGORY_MAP = {
    "O1": "hate_extremism",
    "O2": "violence_gore",
    "O3": "sexual",
    "O4": "nudity_contextual",
    "O5": "criminal_financial",
    "O6": "dangerous",
    "O7": "self_harm_instruction",
    "O8": "personal_financial_data",
    "O9": "brand_reputation",
    "NA": "none",
}


POLICY_VERSION = "bank-image-safety-v1.1"


CATEGORY_METADATA = {
    "sexual": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Pornographic or explicit sexual imagery is not acceptable in bank-owned channels.",
    },
    "sexual_minors": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Sexualized minors are illegal and require fail-closed handling.",
    },
    "nudity_contextual": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Nudity can be legitimate in medical, educational, or artistic context.",
    },
    "violence_gore": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Graphic violence and gore create reputational and user-safety risk.",
    },
    "violence_contextual": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Non-graphic weapons or violence can require contextual interpretation.",
    },
    "dangerous": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Weapons, explosives, drugs, terrorism, and self-harm content can enable real-world harm.",
    },
    "self_harm_instruction": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Actionable self-harm instructions require fail-closed handling.",
    },
    "hate_extremism": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Hate and extremist symbolism are unacceptable for bank distribution.",
    },
    "toxic_inappropriate": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Toxic or inappropriate prompt text can harm user trust and brand reputation.",
    },
    "criminal_financial": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Fraud and criminal planning are directly relevant to banking risk.",
    },
    "personal_financial_data": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Personal and financial data disclosure creates privacy and compliance risk.",
    },
    "brand_reputation": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Unsafe use of bank branding can require manual legal or brand review.",
    },
    "composite_violation": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Prompt, input image, and output image may jointly violate policy.",
    },
    "sexual_pornography": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Pornography and explicit sexual content are not acceptable in bank-owned channels.",
    },
    "extremism_terrorism_symbols": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Extremism, terrorism, and prohibited symbols create legal and reputational risk.",
    },
    "violence_cruelty_gore": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Graphic violence, cruelty, blood, and torture create user-safety and reputational risk.",
    },
    "self_harm_suicide": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Self-harm and suicide content require fail-closed handling.",
    },
    "drugs_propaganda": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Drug-related promotion or depiction can create legal and reputational risk.",
    },
    "weapons_explosives_crime_instructions": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Weapons, explosives, and criminal instructions can enable real-world harm.",
    },
    "fraud_phishing_social_engineering": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Fraud, phishing, and social engineering are directly relevant to banking abuse.",
    },
    "fake_documents_cards_contracts": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Fake documents, cards, certificates, and contracts can enable fraud.",
    },
    "personal_biometric_data": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Personal and biometric data disclosure creates privacy and compliance risk.",
    },
    "misleading_financial_content": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Misleading financial content can harm clients and violate financial advertising rules.",
    },
    "market_investment_manipulation": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Investment manipulation and market abuse can create regulatory risk.",
    },
    "discrimination_hate_speech": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Discrimination and hate speech are unacceptable in bank channels.",
    },
    "political_agitation_symbols": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Political agitation and disputed symbols require contextual legal review.",
    },
    "brand_ip_misuse": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Unauthorized use of brands, logos, and intellectual property can create legal risk.",
    },
    "official_interface_impersonation": {
        "severity": "high",
        "default_action": "block",
        "regulatory_rationale": "Imitation of official interfaces or government bodies can enable deception.",
    },
    "bank_reputation_harm": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Content defaming the bank, clients, or employees requires reputation review.",
    },
    "gambling_casino_quick_money": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Gambling, betting, casinos, and quick-money claims can create compliance and reputation risk.",
    },
    "sanctions_military_geopolitical": {
        "severity": "medium",
        "default_action": "review",
        "regulatory_rationale": "Sanctions, military, and geopolitically risky content require contextual review.",
    },
    "fraudulent_qr_payment_push": {
        "severity": "critical",
        "default_action": "block",
        "regulatory_rationale": "Fraudulent QR codes, payment forms, and fake push notifications can enable payment fraud.",
    },
}


PROMPT_KEYWORDS = {
    "sexual_pornography": [
        "porn",
        "pornographic",
        "explicit sex",
        "эротика",
        "порно",
        "обнажен",
        "сексуальный контент",
        "порнограф",
        "интимное фото",
    ],
    "sexual_minors": [
        "sexual minor",
        "underage sexual",
        "child sexual",
        "несовершеннолет",
        "малолет",
        "детское порно",
        "школьница интим",
    ],
    "extremism_terrorism_symbols": [
        "nazi propaganda",
        "terrorist poster",
        "draw a swastika",
        "swastika",
        "nazi symbol",
        "ss symbol",
        "extremist symbol",
        "экстремистский плакат",
        "экстремизм",
        "терроризм",
        "террорист",
        "запрещенная символика",
        "запрещенная символика",
        "нацистская пропаганда",
        "свастик",
        "нацистский символ",
        "символ сс",
        "зиг хайль",
    ],
    "violence_cruelty_gore": [
        "gore",
        "blood everywhere",
        "severed",
        "torture",
        "graphic violence",
        "cruelty",
        "расчлен",
        "кровища",
        "графическое насилие",
        "кровь",
        "пытк",
        "жестокость",
        "шок контент",
    ],
    "self_harm_suicide": [
        "self-harm",
        "suicide",
        "kill myself",
        "самоповрежд",
        "суицид",
        "самоубийств",
        "порезать вены",
    ],
    "drugs_propaganda": [
        "draw drugs",
        "illegal drugs",
        "drug use",
        "narcotics",
        "наркотик",
        "наркоту",
        "запрещенные вещества",
        "марихуан",
        "кокаин",
        "героин",
        "метамфетамин",
        "пропаганда наркот",
    ],
    "weapons_explosives_crime_instructions": [
        "bomb making",
        "build a bomb",
        "explosive recipe",
        "weapon instructions",
        "crime instructions",
        "как сделать бомбу",
        "самодельное оружие",
        "оружие",
        "взрывчат",
        "как вскрыть банкомат",
        "ограбить банк",
        "криминальная инструкция",
    ],
    "fraud_phishing_social_engineering": [
        "phishing",
        "social engineering",
        "scam",
        "мошеннич",
        "фишинг",
        "социальная инженерия",
        "обмануть клиента",
        "украсть пароль",
    ],
    "fake_documents_cards_contracts": [
        "fake passport",
        "fake bank card",
        "fake contract",
        "поддельный паспорт",
        "поддельная карта",
        "поддельная справка",
        "поддельный договор",
        "нарисуй справку 2 ндфл",
        "фальшивый документ",
    ],
    "personal_biometric_data": [
        "card number",
        "bank account number",
        "passport scan",
        "biometric data",
        "face id",
        "fingerprint",
        "номер карты",
        "паспортные данные",
        "номер счета",
        "персональные данные",
        "биометрические данные",
        "отпечаток пальца",
        "скан лица",
        "cvv",
        "cvc",
        "снилс",
        "инн",
    ],
    "misleading_financial_content": [
        "guaranteed income",
        "risk-free profit",
        "гарантированный доход",
        "безрисковая прибыль",
        "вложи и точно заработаешь",
        "финансово вводящий в заблуждение",
    ],
    "market_investment_manipulation": [
        "pump and dump",
        "market manipulation",
        "инвестиционная манипуляция",
        "манипуляция рынком",
        "разогнать акцию",
        "памп и дамп",
    ],
    "discrimination_hate_speech": [
        "hate speech",
        "racial slur",
        "дискриминац",
        "разжигание ненависти",
        "hate speech",
        "унижение национальности",
    ],
    "political_agitation_symbols": [
        "political agitation",
        "campaign poster",
        "предвыборная агитация",
        "политическая агитация",
        "спорная политическая символика",
        "агитационный плакат",
    ],
    "brand_ip_misuse": [
        "use logo without permission",
        "copyright infringement",
        "незаконное использование бренда",
        "чужой логотип",
        "логотип альфа банка",
        "интеллектуальная собственность",
    ],
    "official_interface_impersonation": [
        "fake government website",
        "fake official interface",
        "имитация госуслуг",
        "фейковый интерфейс банка",
        "поддельный сайт госоргана",
        "официальный интерфейс",
    ],
    "bank_reputation_harm": [
        "defame the bank",
        "банк мошенники",
        "порочащий банк",
        "порочащий клиентов",
        "порочащий сотрудников",
        "унижение сотрудников банка",
    ],
    "gambling_casino_quick_money": [
        "casino",
        "betting",
        "quick money",
        "казино",
        "ставки",
        "азартные игры",
        "быстрые деньги",
        "легкий заработок",
    ],
    "sanctions_military_geopolitical": [
        "sanctions evasion",
        "military propaganda",
        "обход санкций",
        "санкционный контент",
        "военная пропаганда",
        "геополитический конфликт",
    ],
    "fraudulent_qr_payment_push": [
        "fake qr payment",
        "fake push notification",
        "fraudulent payment form",
        "мошеннический qr",
        "фейковый qr код",
        "фейковое push уведомление",
        "поддельная платежная форма",
        "фейковая оплата",
    ],
}


def _load_external_policy() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


_external_policy = _load_external_policy()
if _external_policy:
    POLICY_VERSION = _external_policy.get("policy_version", POLICY_VERSION)
    CATEGORY_METADATA = {
        **CATEGORY_METADATA,
        **(_external_policy.get("categories") or {}),
    }
    PROMPT_KEYWORDS = {
        **PROMPT_KEYWORDS,
        **{
            category: sorted(set(PROMPT_KEYWORDS.get(category, []) + keywords))
            for category, keywords in (_external_policy.get("prompt_keywords") or {}).items()
        },
    }
