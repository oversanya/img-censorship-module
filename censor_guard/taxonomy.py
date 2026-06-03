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
SAFE_VISUAL_LABEL = "safe everyday image"
VISUAL_LABELS = [spec.label for spec in CATEGORY_SPECS] + [SAFE_VISUAL_LABEL]
HARD_BLOCK_CATEGORIES = {spec.code for spec in CATEGORY_SPECS if spec.hard_block}
SOFT_REVIEW_CATEGORIES = {spec.code for spec in CATEGORY_SPECS if not spec.hard_block}


def is_known_category(code: str) -> bool:
    return code in CATEGORY_BY_CODE

