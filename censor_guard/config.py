from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _default_hf_cache() -> str:
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / ".cache" / "huggingface")


def _default_tesseract_cmd() -> str | None:
    discovered = shutil.which("tesseract")
    if discovered:
        return discovered

    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        project_root / "tools" / "Tesseract-OCR" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path("/opt/homebrew/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
        Path("/usr/bin/tesseract"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _default_tessdata_dir() -> str | None:
    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        project_root / "tools" / "tessdata",
        project_root / "tools" / "Tesseract-OCR" / "tessdata",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    tesseract_cmd = _default_tesseract_cmd()
    if tesseract_cmd:
        candidate = Path(tesseract_cmd).parent / "tessdata"
        if candidate.exists():
            return str(candidate)
    return None


@dataclass(frozen=True)
class Settings:
    enable_ocr: bool = _env_bool("CENSOR_ENABLE_OCR", True)
    enable_visual_classifier: bool = _env_bool("CENSOR_ENABLE_VISUAL_CLASSIFIER", True)
    enable_explicit_detector: bool = _env_bool("CENSOR_ENABLE_EXPLICIT_DETECTOR", True)
    enable_text_guard: bool = _env_bool("CENSOR_ENABLE_TEXT_GUARD", True)
    enable_injection_revealer: bool = _env_bool("CENSOR_ENABLE_INJECTION_REVEALER", True)
    enable_image_sanitizer: bool = _env_bool("CENSOR_ENABLE_IMAGE_SANITIZER", True)
    # Робастный цензор (Linf-robust ConvNeXt + linear probe) как ИНДИКАТОР
    # adversarial-примеров. Сам по себе категории не выставляет — сравнивается с
    # основным пайплайном (см. robust_unsafe_min). Мягко
    # деградирует, если robustbench/веса недоступны.
    enable_robust_guard: bool = _env_bool("CENSOR_ENABLE_ROBUST_GUARD", True)
    # Всегда работающий обученный визуальный safety-судья (LlavaGuard-0.5B). Второй
    # независимый визуальный сенсор рядом с CLIP. На CPU ~10–30 с/картинку — можно
    # выключить флагом, тогда визуал держит только CLIP + NSFW-детектор.
    enable_llava_guard: bool = _env_bool("CENSOR_ENABLE_LLAVA_GUARD", True)
    # Включает эскалацию на ShieldGemma (доп. слой судьи). fusion работает всегда,
    # независимо от этого флага.
    enable_policy_judge: bool = _env_bool("CENSOR_ENABLE_POLICY_JUDGE", False)
    visual_model_id: str = os.getenv("CENSOR_VISUAL_MODEL_ID", "openai/clip-vit-base-patch32")
    explicit_model_id: str = os.getenv("CENSOR_EXPLICIT_MODEL_ID", "Falconsai/nsfw_image_detection")
    # ML-модель текстового гарда (HF text-classification). По умолчанию дешёвая
    # русскоязычная toxicity-модель (~50МБ, быстрая на CPU). Сливается с лексиконом
    # по максимуму. Пустая строка → только лексиконный baseline (без зависимостей).
    text_model_id: str = os.getenv("CENSOR_TEXT_MODEL_ID", "cointegrated/rubert-tiny-toxicity")
    policy_judge_model_id: str = os.getenv("CENSOR_POLICY_JUDGE_MODEL_ID", "google/shieldgemma-2-4b-it")
    llava_guard_model_id: str = os.getenv("CENSOR_LLAVA_GUARD_MODEL_ID", "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf")
    # rating+category стоят первыми в JSON — этого хватает для решения; rationale
    # (поясняющий текст) только удлиняет генерацию (~0.5 с/токен на CPU). Поднять,
    # если нужен развёрнутый rationale в аудите.
    llava_guard_max_new_tokens: int = int(_env_float("CENSOR_LLAVA_GUARD_MAX_NEW_TOKENS", 24))
    # Оценка, которую кладём в категорию при вердикте Unsafe (модель бинарна; силу
    # сигнала дополнительно модулирует вес сенсора в fusion).
    llava_guard_unsafe_score: float = _env_float("CENSOR_LLAVA_GUARD_UNSAFE_SCORE", 0.9)
    block_threshold: float = _env_float("CENSOR_BLOCK_THRESHOLD", 0.85)
    review_threshold: float = _env_float("CENSOR_REVIEW_THRESHOLD", 0.55)
    # Порог калибровки CLIP относительно safe-якоря (равные шансы = 0.5). Ниже —
    # чувствительнее (больше review/block), выше — консервативнее. Дефолт 0.35:
    # CLIP-base слабоват, 0.5 давил его почти в ноль; 0.35 даёт сигнал, оставляя
    # безобидное в нуле, а шум — ниже порога улики 0.3. См. calibration.py.
    calibration_floor: float = _env_float("CENSOR_CALIBRATION_FLOOR", 0.35)
    # Веса linear probe робастного цензора. По умолчанию лежат в корне проекта
    # (robust_probe.npz). Каталог model_dir — куда robustbench кладёт/ищет
    # чекпойнт базовой робастной модели.
    robust_probe_path: str = os.getenv(
        "CENSOR_ROBUST_PROBE_PATH",
        str(Path(__file__).resolve().parent.parent / "models/robust_probe.npz"),
    )
    robust_model_dir: str = os.getenv(
        "CENSOR_ROBUST_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent / "models"),
    )
    # Детектор adversarial = НАПРАВЛЕННОЕ расхождение моделей: блокируем, только если
    # робастная модель уверенно говорит «unsafe» (P ≥ robust_unsafe_min), А основной
    # пайплайн при этом картинку пропускает (unsafe_score < review_threshold). Это
    # ровно сценарий обхода: возмущение обмануло основную модель, но робастную — нет.
    # Симметричный abs-порог тут плох (на чистых картинках расхождение шкал даёт
    # ~27% ложных блокировок); направленное правило при том же отлове adversarial
    # даёт ~4.5% FP. Калибровано на benchmarking-сплите (см. reports/). 0.7 — колено
    # кривой (отлов ~97%); 0.9 — консервативнее (~84% при ~1.7% FP).
    robust_unsafe_min: float = _env_float("CENSOR_ROBUST_UNSAFE_MIN", 0.7)
    hf_cache_dir: str = os.getenv("CENSOR_HF_CACHE_DIR", _default_hf_cache())
    tesseract_cmd: str | None = os.getenv("CENSOR_TESSERACT_CMD", _default_tesseract_cmd() or "")
    tessdata_dir: str | None = os.getenv("CENSOR_TESSDATA_DIR", _default_tessdata_dir() or "")