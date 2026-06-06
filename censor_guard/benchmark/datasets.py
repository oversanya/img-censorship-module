"""Загрузчик курируемого бенчмарк-датасета.

Один источник истины — `vekshinkir/image-censorship-small`, split `benchmarking`
(690 размеченных картинок). Раньше здесь жил реестр из ~14 сетевых датасетов с
пробами/таймаутами; теперь это лишний слой — у нас есть готовый курируемый набор.

`load_benchmark` возвращает плоский список dict'ов «строка = картинка + метаданные».
Картинки приводятся к RGB; битый пример молча пропускается, чтобы один сбой декода
не ронял весь прогон.
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

DATASET_ID = "vekshinkir/image-censorship-small"
DEFAULT_SPLIT = "benchmarking"

# Категория датасета → код таксономии (`censor_guard.taxonomy`). Применяется ТОЛЬКО
# к unsafe-строкам: у safe-картинок `category` — это тип контента (animal/art/food…),
# а не нарушение, поэтому `true_category` у них остаётся None.
CATEGORY_MAP: dict[str, str] = {
    "sexual": "sexual",
    "violence": "violence_gore",
    "gore": "violence_gore",
    "weapons": "illegal_activity",
    "illegal activity": "illegal_activity",
    "self-harm": "self_harm",
    "hate": "hate_extremism",
    "extremism": "hate_extremism",
    "toxic_meme": "hate_extremism",
    "harassment": "harassment",
    "deception": "deception_fraud",
    "political": "political_persuasion",
    "public and personal health": "health_misinformation",
    "spam": "spam_scams",
    "shocking": "shocking",
}


def _to_rgb(value: Any) -> Image.Image | None:
    """Привести значение поля-картинки к PIL RGB. None при неудаче."""
    try:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict) and value.get("bytes"):
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if isinstance(value, str):
            return Image.open(value).convert("RGB")
    except Exception:
        return None
    return None


def load_benchmark(
    split: str = DEFAULT_SPLIT, limit: int | None = None, seed: int = 42,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Загрузить размеченные картинки из курируемого датасета.

    Args:
        split: сплит датасета (по умолчанию `benchmarking`).
        limit: взять не более N картинок (после перемешивания); None → все.
        seed: сид перемешивания.
        token: HF-токен (если датасет приватный); None → анонимно/из кэша.

    Returns:
        Список dict'ов с ключами `image` (RGB) + метаданные: `label` (0/1),
        `category`, `subcategory`, `is_edge_case`, `ai_generated`, `source`, `text`.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, token=token)
    label_names = getattr(ds.features.get("label"), "names", None)  # ['safe','unsafe']

    order = list(range(ds.num_rows))
    random.Random(seed).shuffle(order)
    if limit is not None:
        order = order[:limit]

    rows: list[dict[str, Any]] = []
    for i in tqdm(order, desc=f"Загрузка {split}", unit="img"):
        ex = ds[i]
        img = _to_rgb(ex.get("image"))
        if img is None:
            continue
        raw = ex.get("label")
        label = label_names.index(raw) if (label_names and isinstance(raw, str)) else int(raw)
        rows.append({
            "image": img,
            "label": int(label),  # 0 safe / 1 unsafe
            "category": ex.get("category"),
            "subcategory": ex.get("subcategory"),
            "is_edge_case": bool(ex.get("is_edge_case")),
            "ai_generated": bool(ex.get("ai_generated")),
            "source": ex.get("source") or "unknown",
            "text": ex.get("text"),
        })
    return rows
