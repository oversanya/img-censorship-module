"""Report generator — produces JSON and Markdown reports for each verdict."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from censorship.core.verdict import Verdict

logger = logging.getLogger(__name__)

# Decision labels in Russian for human-readable reports
_DECISION_LABELS_RU = {
    "ALLOW": "РАЗРЕШЕНО",
    "BLOCK": "ЗАБЛОКИРОВАНО",
    "REVIEW": "НА ПРОВЕРКУ",
}

_CATEGORY_LABELS_RU = {
    "sexual_explicit": "Сексуальный контент",
    "violence_gore": "Насилие / Жестокость",
    "extremism": "Экстремизм / Терроризм",
    "hate_speech": "Язык ненависти",
    "personal_data": "Персональные данные / ПДн",
    "financial_fraud": "Финансовое мошенничество",
    "csam": "CSAM (эксплуатация несовершеннолетних)",
}


@dataclass
class ReportFiles:
    json_path: Path
    markdown_path: Path
    heatmap_path: Path | None = None
    attention_path: Path | None = None

    def __str__(self) -> str:
        parts = [f"JSON: {self.json_path}", f"Markdown: {self.markdown_path}"]
        if self.heatmap_path:
            parts.append(f"Heatmap: {self.heatmap_path}")
        if self.attention_path:
            parts.append(f"Attention: {self.attention_path}")
        return " | ".join(parts)


class ReportGenerator:
    """Generates machine-readable JSON and human-readable Markdown reports."""

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        verdict: Verdict,
        image_path: Union[str, Path] | None = None,
        heatmap_path: Path | None = None,
        attention_path: Path | None = None,
    ) -> ReportFiles:
        """Generate both JSON and Markdown reports."""
        short_id = verdict.image_id[:12]
        json_path = self._write_json(verdict, short_id, heatmap_path, attention_path)
        md_path = self._write_markdown(verdict, short_id, image_path, heatmap_path, attention_path)
        return ReportFiles(
            json_path=json_path,
            markdown_path=md_path,
            heatmap_path=heatmap_path,
            attention_path=attention_path,
        )

    def _write_json(
        self,
        verdict: Verdict,
        short_id: str,
        heatmap_path: Path | None,
        attention_path: Path | None,
    ) -> Path:
        """Machine-readable JSON for SIEM/audit systems."""
        report = {
            "schema_version": "1.0",
            "image_id": f"sha256:{verdict.image_id}",
            "timestamp": verdict.timestamp,
            "decision": verdict.decision,
            "primary_category": verdict.primary_category,
            "confidence": round(verdict.reasoner_confidence or 0.0, 4),
            "classifier": verdict.classifier_model,
            "classifier_scores": {
                k: round(v, 4) for k, v in verdict.classifier_scores.items()
            },
            "classifier_triggered": verdict.classifier_triggered,
            "reasoner": verdict.reasoner_model,
            "rationale": verdict.reasoner_rationale,
            "explanation_for_user": verdict.explanation_for_user,
            "explanation_for_regulator": verdict.explanation_for_regulator,
            "prompt_verdict": verdict.prompt_verdict,
            "prompt_category": verdict.prompt_category,
            "latency_ms": round(verdict.latency_ms, 2),
            "pipeline_version": verdict.pipeline_version,
            "heatmap_path": str(heatmap_path) if heatmap_path else None,
            "attention_path": str(attention_path) if attention_path else None,
        }
        output_path = self.output_dir / f"{short_id}_report.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON report saved to {output_path}")
        return output_path

    def _write_markdown(
        self,
        verdict: Verdict,
        short_id: str,
        image_path: Union[str, Path] | None,
        heatmap_path: Path | None,
        attention_path: Path | None,
    ) -> Path:
        """Human-readable Markdown for regulators and analysts."""
        decision_ru = _DECISION_LABELS_RU.get(verdict.decision, verdict.decision)
        category_ru = _CATEGORY_LABELS_RU.get(
            verdict.primary_category or "", verdict.primary_category or "—"
        )
        confidence = round((verdict.reasoner_confidence or 0.0) * 100, 1)

        # Build score table
        score_rows = ""
        for cat, score in sorted(verdict.classifier_scores.items(), key=lambda x: -x[1]):
            cat_ru = _CATEGORY_LABELS_RU.get(cat, cat)
            pct = round(score * 100, 1)
            flag = " ⚠️" if score >= 0.50 else ""
            score_rows += f"| {cat_ru} | {pct}%{flag} |\n"

        # Heatmap section
        heatmap_section = ""
        if heatmap_path and Path(heatmap_path).exists():
            heatmap_section = f"\n## Тепловая карта\n![Heatmap]({heatmap_path})\n"
        if attention_path and Path(attention_path).exists():
            heatmap_section += f"\n## Карта внимания модели\n![Attention]({attention_path})\n"

        # Image section
        image_section = ""
        if image_path and Path(image_path).exists():
            image_section = f"\n**Файл изображения:** `{image_path}`\n"

        md = f"""# Заключение по изображению `{short_id}...`

**Решение:** {decision_ru}
**Категория:** {category_ru}
**Уверенность:** {confidence}%
**Время обработки:** {round(verdict.latency_ms, 1)} мс
**Версия пайплайна:** {verdict.pipeline_version}
**Метка времени:** {verdict.timestamp}
{image_section}
---

## Оценки по категориям

| Категория | Уверенность |
|-----------|-------------|
{score_rows}
---

## Обоснование

{verdict.reasoner_rationale or "Нет дополнительного обоснования."}

---

## Объяснение для пользователя

{verdict.explanation_for_user or "—"}

---

## Формальное заключение (для регулятора)

{verdict.explanation_for_regulator or "—"}

---

## Технические детали

| Параметр | Значение |
|----------|---------|
| Классификатор (Слой 1) | `{verdict.classifier_model}` |
| Обоснователь (Слой 2) | `{verdict.reasoner_model or "не использовался"}` |
| ID изображения | `sha256:{verdict.image_id}` |
| Результат проверки промпта | {verdict.prompt_verdict or "—"} |
| Категория промпта | {verdict.prompt_category or "—"} |
{heatmap_section}
---
*Сгенерировано автоматически системой цензур-модуля банка. Pipeline v{verdict.pipeline_version}.*
"""
        output_path = self.output_dir / f"{short_id}_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"Markdown report saved to {output_path}")
        return output_path
