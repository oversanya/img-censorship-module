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


POLICY_VERSION = "bank-image-safety-v1.0"


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
}


PROMPT_KEYWORDS = {
    "sexual": [
        "porn",
        "pornographic",
        "explicit sex",
        "эротика",
        "порно",
        "обнажен",
    ],
    "violence_gore": [
        "gore",
        "blood everywhere",
        "severed",
        "расчлен",
        "кровища",
    ],
    "dangerous": [
        "bomb making",
        "build a bomb",
        "explosive recipe",
        "draw drugs",
        "illegal drugs",
        "drug use",
        "narcotics",
        "как сделать бомбу",
        "самодельное оружие",
        "наркотик",
        "наркоту",
        "запрещенные вещества",
        "марихуан",
        "кокаин",
        "героин",
        "метамфетамин",
        "суицид",
        "самоубийств",
        "как вскрыть банкомат",
        "ограбить банк",
    ],
    "hate_extremism": [
        "nazi propaganda",
        "terrorist poster",
        "draw a swastika",
        "swastika",
        "nazi symbol",
        "ss symbol",
        "экстремистский плакат",
        "нацистская пропаганда",
        "свастик",
        "нацистский символ",
        "символ сс",
        "зиг хайль",
    ],
    "personal_financial_data": [
        "card number",
        "bank account number",
        "passport scan",
        "номер карты",
        "паспортные данные",
        "номер счета",
        "cvv",
        "cvc",
        "снилс",
        "инн",
    ],
}
