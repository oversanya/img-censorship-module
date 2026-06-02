"""Taxonomy loader — reads and validates taxonomy.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


Priority = Literal["critical", "high", "medium", "low"]


class CategoryConfig(BaseModel):
    id: str
    label: str
    priority: Priority
    threshold_block: float = 0.80
    threshold_review: float = 0.50
    description: str = ""

    @field_validator("threshold_block", "threshold_review")
    @classmethod
    def _valid_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be in [0, 1], got {v}")
        return v


class TaxonomyLoader:
    def __init__(self, taxonomy_path: str | Path):
        self._path = Path(taxonomy_path)
        self._categories: dict[str, CategoryConfig] = {}
        self._load()

    def _load(self) -> None:
        with open(self._path) as f:
            data = yaml.safe_load(f)
        for raw in data["categories"]:
            cfg = CategoryConfig(**raw)
            self._categories[cfg.id] = cfg

    def get(self, category_id: str) -> CategoryConfig | None:
        return self._categories.get(category_id)

    def all(self) -> list[CategoryConfig]:
        return list(self._categories.values())

    def ids(self) -> list[str]:
        return list(self._categories.keys())

    def is_critical(self, category_id: str) -> bool:
        cfg = self._categories.get(category_id)
        return cfg is not None and cfg.priority == "critical"
