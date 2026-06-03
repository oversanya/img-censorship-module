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
        label="sexual or explicit adult content",
        description="nudity, pornography, or sexualized adult imagery",
        hard_block=True,
    ),
    CategorySpec(
        code="sexual_minors",
        label="sexual content involving minors",
        description="sexualized minors, child exploitation, or grooming context",
        hard_block=True,
    ),
    CategorySpec(
        code="violence_gore",
        label="graphic violence gore blood or torture",
        description="physical violence, gore, blood, torture, or mutilation",
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
        label="extremism terrorism prohibited symbols",
        description="terrorist content, extremist propaganda, prohibited symbols, or calls to violence",
        hard_block=True,
    ),
    CategorySpec(
        code="drugs",
        label="drugs narcotics propaganda",
        description="drug production, sale, use, or promotion",
        hard_block=True,
    ),
    CategorySpec(
        code="weapons_crime",
        label="weapons explosives criminal instructions",
        description="weapons, explosives, or operational criminal instructions",
        hard_block=True,
    ),
    CategorySpec(
        code="deception_fraud",
        label="fraud phishing social engineering",
        description="phishing, scams, social engineering, or deceptive financial collection",
        hard_block=True,
    ),
    CategorySpec(
        code="forged_documents",
        label="forged documents cards contracts",
        description="fake passports, certificates, payment cards, contracts, or bank documents",
        hard_block=True,
    ),
    CategorySpec(
        code="personal_biometric_data",
        label="personal biometric confidential data",
        description="passports, phone numbers, card details, faces, fingerprints, or other sensitive identity data",
        hard_block=True,
    ),
    CategorySpec(
        code="fraudulent_qr_payment",
        label="fraudulent qr codes payment forms fake push notifications",
        description="fraudulent QR codes, payment forms, bank login screens, or fake push notifications",
        hard_block=True,
    ),
    CategorySpec(
        code="harassment",
        label="harassment or abuse",
        description="targeted insults, humiliation, or threatening abuse",
        hard_block=False,
    ),
    CategorySpec(
        code="financial_misleading",
        label="financially misleading content",
        description="misleading lending, deposit, cashback, or other financial claims",
        hard_block=False,
    ),
    CategorySpec(
        code="investment_manipulation",
        label="investment market manipulation",
        description="pump and dump, guaranteed profit claims, or manipulative investment creatives",
        hard_block=False,
    ),
    CategorySpec(
        code="discrimination_hate",
        label="discrimination hate speech",
        description="discriminatory, hateful, or dehumanizing content toward protected groups",
        hard_block=False,
    ),
    CategorySpec(
        code="political_persuasion",
        label="political persuasion controversial political symbolism",
        description="political campaigning, election influence, or controversial political symbols",
        hard_block=False,
    ),
    CategorySpec(
        code="brand_ip_abuse",
        label="illegal brand logo intellectual property use",
        description="unauthorized brand, logo, trademark, or copyrighted asset use",
        hard_block=False,
    ),
    CategorySpec(
        code="official_interface_impersonation",
        label="official interface government or bank impersonation",
        description="fake government portals, bank interfaces, official notices, or regulator pages",
        hard_block=False,
    ),
    CategorySpec(
        code="bank_reputation_risk",
        label="content defaming bank clients or employees",
        description="content that may defame the bank, clients, employees, or partners",
        hard_block=False,
    ),
    CategorySpec(
        code="gambling_fast_money",
        label="gambling betting casino fast money",
        description="casino, betting, lotteries, or get-rich-quick promotions",
        hard_block=False,
    ),
    CategorySpec(
        code="sanctions_geopolitical",
        label="sanctions military geopolitical risk",
        description="sanctioned entities, military propaganda, or high-risk geopolitical content",
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
SAFE_VISUAL_LABEL = "safe everyday image"
VISUAL_LABELS = [spec.label for spec in CATEGORY_SPECS] + [SAFE_VISUAL_LABEL]
HARD_BLOCK_CATEGORIES = {spec.code for spec in CATEGORY_SPECS if spec.hard_block}
SOFT_REVIEW_CATEGORIES = {spec.code for spec in CATEGORY_SPECS if not spec.hard_block}


def is_known_category(code: str) -> bool:
    return code in CATEGORY_BY_CODE
