import re
from typing import List


LEET_TABLE = str.maketrans(
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


def normalize_text_variants(text: str) -> List[str]:
    lowered = text.lower().replace("ё", "е")
    leet = lowered.translate(LEET_TABLE)
    compact = compact_text(lowered)
    compact_leet = compact_text(leet)
    collapsed = re.sub(r"\s+", " ", lowered).strip()

    variants = [lowered, collapsed, leet, compact, compact_leet]
    return list(dict.fromkeys(variant for variant in variants if variant))


def compact_text(text: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", text.lower().replace("ё", "е"))

