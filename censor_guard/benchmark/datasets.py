"""Реестр бенчмарк-датасетов и устойчивые загрузчики-сэмплеры.

Каждый датасет из `censor_benchmark_datasets.md` описан как :class:`DatasetSpec`
с привязкой к коду нашей таксономии (`censor_guard.taxonomy`). Загрузчик
`sample_dataset` возвращает список `(PIL.Image RGB, label)` фиксированного размера
и НИКОГДА не роняет бенчмарк: любая недоступность (нет `HF_TOKEN`, нет
Kaggle-ключей, сетевой обрыв, gated-доступ) превращается в «мягкий» пропуск с
человекочитаемой причиной (`SampleResult.skipped_reason`).

Стриминговые наборы тянутся устойчивым циклом с ретраями и докачкой через
`.skip()` (паттерн перенесён из `notebooks/metrics_benchmark.ipynb`).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from PIL import Image

from censor_guard.evaluation import UNSAFEBENCH_TO_TAXONOMY


# ─────────────────────────────────────────────────────────────────────────────
# Метка одной картинки в бенчмарке.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Label:
    """Истинная разметка одной картинки.

    `true_unsafe` = None означает, что у датасета нет бинарной метки на уровне
    картинки (такого в текущем реестре нет, но интерфейс это допускает).
    `adversarial` помечает hard-negative/обфускацию — такие строки считаются
    отдельно и исключаются из общих метрик.
    """

    true_unsafe: bool | None
    true_category: str | None = None
    adversarial: bool = False


# Тип лейблера: по сырому примеру датасета возвращает Label либо None
# (None = пример нужно пропустить, например out-of-scope метка).
Labeler = Callable[[dict[str, Any]], "Label | None"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # короткий идентификатор для CLI/отчёта
    hf_id: str  # repo id на HuggingFace (или псевдо-id для не-HF источников)
    category: str  # код таксономии или "safe"
    kind: str  # "all_unsafe" | "all_safe" | "binary"
    access: str = "public"  # "public" | "hf_token" | "kaggle" | "request"
    config: str | None = None
    split: str = "train"
    streaming: bool = False
    image_fields: tuple[str, ...] = ("image", "pixel_values", "img", "jpg", "png")
    label_fields: tuple[str, ...] = ()
    # Значения метки (после нормализации в lower-str), означающие «небезопасно».
    positive_values: tuple[str, ...] = ()
    # per_class: сэмплировать до `n` примеров на КАЖДУЮ ячейку (категория×label),
    # а не n суммарно. Используется для UnsafeBench, чтобы плотно покрыть все
    # категории таксономии.
    per_class: bool = False
    note: str = ""

    @property
    def title(self) -> str:
        cfg = f" [{self.config}]" if self.config else ""
        return f"{self.hf_id}{cfg}"

    @property
    def gated_url(self) -> str | None:
        """Ссылка на страницу принятия условий для gated-датасетов."""
        if self.access == "hf_token":
            return f"https://huggingface.co/datasets/{self.hf_id}"
        return None


@dataclass
class SampleResult:
    spec: DatasetSpec
    items: list[tuple[Image.Image, Label]] = field(default_factory=list)
    skipped_reason: str | None = None
    load_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None and len(self.items) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Реестр датасетов (максимальный доступный набор).
#
# Категории, для которых НЕТ выделенного публичного image-датасета
# (self_harm, illegal_activity, shocking, deception_fraud, political_persuasion,
# health_misinformation, spam_scams, harassment), покрываются через UnsafeBench —
# он несёт поле `category` со всеми 11 классами.
# ─────────────────────────────────────────────────────────────────────────────
REGISTRY: tuple[DatasetSpec, ...] = (
    # ── UnsafeBench — backbone метрик по категориям (safe + unsafe, 11 классов) ──
    DatasetSpec(
        key="unsafebench",
        hf_id="yiting/UnsafeBench",
        category="multi",
        kind="binary",
        split="test",
        label_fields=("safety_label",),
        positive_values=("unsafe",),
        per_class=True,  # до n примеров на КАЖДУЮ ячейку (категория × safe/unsafe)
        note="11 категорий + safe; источник per-category метрик",
    ),
    # ── SEXUAL ───────────────────────────────────────────────────────────────
    DatasetSpec(
        key="nsfw1024",
        hf_id="yesidobyte/nsfw1024",
        category="sexual",
        kind="all_unsafe",
        streaming=True,
    ),
    DatasetSpec(
        key="civitai_nsfw",
        hf_id="wallstoneai/civitai-top-nsfw-images-with-metadata",
        category="sexual",
        kind="all_unsafe",
        streaming=True,
        note="AI-генерация (SD/Flux)",
    ),
    DatasetSpec(
        key="deepghs_nsfw",
        hf_id="deepghs/nsfw_detect",
        category="sexual",
        kind="binary",
        access="hf_token",
        streaming=True,
        label_fields=("label",),
        positive_values=("porn", "hentai", "sexy"),
        note="🔒 gated: multiclass porn/hentai/sexy→unsafe, neutral/drawing→safe",
    ),
    # ── VIOLENCE / GORE ──────────────────────────────────────────────────────
    DatasetSpec(
        key="violence_unsafe",
        hf_id="Lenkashell/unsafe_violence_image_captions",
        category="violence_gore",
        kind="all_unsafe",
        streaming=True,
        note="реальные сцены насилия",
    ),
    DatasetSpec(
        key="violence_safe",
        hf_id="Lenkashell/safe_violence_image_captions",
        category="safe",
        kind="all_safe",
        streaming=True,
        note="safe-пара к violence (контроль FPR)",
    ),
    DatasetSpec(
        key="gore_blood",
        hf_id="NeuralShell/Gore-Blood-Dataset-v1.0",
        category="violence_gore",
        kind="all_unsafe",
        access="hf_token",
        note="🔒 gated: кровь/gore",
    ),
    # ── HATE / EXTREMISM (мемы с реальными картинками) ───────────────────────
    DatasetSpec(
        key="memelens_ru",
        hf_id="QCRI/MemeLens",
        config="toxic_ru__Toxic_Memes_Detection_Dataset",
        category="hate_extremism",
        kind="binary",
        streaming=True,
        label_fields=("label",),
        positive_values=("toxic", "hateful", "harmful", "1"),
        note="русские токсичные мемы",
    ),
    DatasetSpec(
        key="memelens_fhm",
        hf_id="QCRI/MemeLens",
        config="Hateful_en_FHM",
        category="hate_extremism",
        kind="binary",
        streaming=True,
        label_fields=("label",),
        positive_values=("hateful", "toxic", "harmful", "1"),
        note="Facebook Hateful Memes (EN)",
    ),
    # ── SAFE-негативы ────────────────────────────────────────────────────────
    DatasetSpec(key="tiny_imagenet", hf_id="zh-plus/tiny-imagenet", category="safe", kind="all_safe", split="valid", streaming=True,
                note="бытовые сцены"),
    DatasetSpec(key="wikiart", hf_id="Artificio/WikiArt", category="safe", kind="all_safe", streaming=True,
                note="классика, в т.ч. ню — edge case"),
    DatasetSpec(key="skin_lesion", hf_id="ahmed-ai/skin-lesions-classification-dataset", category="safe", kind="all_safe", streaming=True,
                note="дерматология — медицинский edge case"),
    DatasetSpec(key="chest_xray", hf_id="hf-vision/chest-xray-pneumonia", category="safe", kind="all_safe", split="train", streaming=True,
                note="рентген грудной клетки — медицинский edge case"),
)

REGISTRY_BY_KEY = {spec.key: spec for spec in REGISTRY}


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательное.
# ─────────────────────────────────────────────────────────────────────────────
def _first_present(example: dict[str, Any], candidates: Iterable[str]) -> Any | None:
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _extract_image(example: dict[str, Any], candidates: Iterable[str]) -> Image.Image | None:
    """Достать картинку из примера, устойчиво к битым/неподдерживаемым форматам.

    Доступ к полю-картинке у `datasets` ленивый: декод происходит при чтении
    `example[key]` и может бросить (напр. WEBP без libwebp, повреждённый файл).
    Любую такую ошибку глотаем и пробуем следующее поле — один битый пример не
    должен ронять загрузку всего датасета.
    """
    for key in candidates:
        if key not in example:
            continue
        try:
            value = example[key]
        except Exception:
            continue
        if value is None:
            continue
        img = _to_rgb(value)
        if img is not None:
            return img
    return None


def _to_rgb(value: Any) -> Image.Image | None:
    """Привести значение поля-картинки к PIL RGB. Возвращает None при неудаче."""
    try:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        # datasets иногда отдаёт dict {'bytes':..., 'path':...} или путь.
        if isinstance(value, dict) and value.get("bytes"):
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if isinstance(value, str):
            return Image.open(value).convert("RGB")
    except Exception:
        return None
    return None


def _normalize_label_value(feature: Any, raw: Any) -> str:
    """Нормализовать сырое значение метки в lower-str.

    Для ClassLabel-фич переводит int → имя класса, иначе str(raw).
    """
    try:
        from datasets import ClassLabel

        if isinstance(feature, ClassLabel) and isinstance(raw, int):
            return feature.int2str(raw).strip().lower()
    except Exception:
        pass
    return str(raw).strip().lower()


def _build_labeler(spec: DatasetSpec, features: Any) -> Labeler:
    """Сконструировать лейблер под конкретный датасет/схему."""
    if spec.kind == "all_unsafe":
        cat = spec.category
        return lambda ex: Label(True, cat, adversarial=False)

    if spec.kind == "all_safe":
        adv = spec.key == "meme_sanity"
        return lambda ex: Label(False, None, adversarial=adv)

    # kind == "binary"
    positives = {v.lower() for v in spec.positive_values}

    def labeler(ex: dict[str, Any]) -> Label | None:
        raw = _first_present(ex, spec.label_fields)
        if raw is None:
            return None
        field_name = next((f for f in spec.label_fields if f in ex), None)
        feat = features.get(field_name) if features is not None and field_name else None
        norm = _normalize_label_value(feat, raw)
        is_unsafe = norm in positives
        if spec.key == "unsafebench":
            # Категорию берём из исходного поля и маппим в нашу таксономию.
            cat_raw = ex.get("category")
            cat = UNSAFEBENCH_TO_TAXONOMY.get(str(cat_raw)) if cat_raw else None
            return Label(is_unsafe, cat if is_unsafe else None)
        return Label(is_unsafe, spec.category if is_unsafe else None)

    return labeler


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка/сэмплирование.
# ─────────────────────────────────────────────────────────────────────────────
def _hf_token(access: str, token: str | None) -> str | None:
    return token if access == "hf_token" else None


def _sample_streaming(spec, n, seed, time_budget, token, progress, max_retries=4):
    """n картинок стримингом — устойчиво к обрывам (ретраи + .skip) и без
    скачивания всех шардов: читаем поток и останавливаемся, как только набрали n.

    Для `kind == "binary"` балансируем на лету: целимся в ~n/2 на класс, чтобы
    в метриках были и safe, и unsafe. Если редкий класс не наберётся в рамках
    тайм-бюджета — берём что есть, бенчмарк не падает.
    """
    from datasets import load_dataset

    balanced = spec.kind == "binary"
    per_class = max(1, n // 2)
    buckets: dict[bool, list[tuple[Image.Image, Label]]] = {True: [], False: []}
    items: list[tuple[Image.Image, Label]] = []

    def total() -> int:
        return (len(buckets[True]) + len(buckets[False])) if balanced else len(items)

    def keep(img, label) -> bool:
        """Решить, берём ли пример (с учётом балансировки), и сохранить."""
        if not balanced:
            items.append((img, label))
            return True
        b = bool(label.true_unsafe)
        if len(buckets[b]) >= per_class and len(buckets[not b]) < per_class:
            return False  # этот класс уже добран, ждём другой
        buckets[b].append((img, label))
        return True

    # Ранний выход на «мёртвых» источниках: если просмотрели уже много сырых
    # примеров, но не набрали НИ одной пригодной картинки (path-only зеркало без
    # байтов, несовместимая схема) — быстро сдаёмся. Считаем именно по числу
    # ПРОСМОТРЕННЫХ примеров, а не по времени: медленная загрузка первого шарда
    # не должна приводить к ложному выходу до появления первой картинки.
    early_bail_seen = 400

    t0 = time.time()
    attempt = 0
    labeler = None
    seen = 0  # сколько сырых примеров уже прошли (для .skip при ретрае)
    while total() < n and attempt <= max_retries and time.time() - t0 < time_budget:
        attempt += 1
        try:
            ds = load_dataset(
                spec.hf_id, spec.config, split=spec.split,
                streaming=True, token=_hf_token(spec.access, token),
            )
            if labeler is None:
                labeler = _build_labeler(spec, getattr(ds, "features", None))
            if seen:
                ds = ds.skip(seen)
            it = iter(ds)
            while True:
                try:
                    ex = next(it)
                except StopIteration:
                    break
                except Exception:
                    continue  # битый пример при чтении потока — пропускаем
                seen += 1
                if time.time() - t0 > time_budget or (total() == 0 and seen >= early_bail_seen):
                    break
                img = _extract_image(ex, spec.image_fields)
                if img is None:
                    continue
                label = labeler(ex)
                if label is None or label.true_unsafe is None:
                    continue
                if keep(img, label) and progress is not None:
                    progress(1)
                if total() >= n:
                    break
            break  # поток исчерпан / набрали n / вышли по бюджету — без исключения
        except Exception as exc:  # сетевой обрыв / gated — ретраим или сдаёмся
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
            if attempt > max_retries:
                if total() == 0:
                    raise RuntimeError(last) from exc
                break
            time.sleep(2)

    if balanced:
        items = buckets[True] + buckets[False]
        random.Random(seed).shuffle(items)
    return items


def _sample_indexed(spec, n, seed, token, progress):
    """Картинки из индексируемого набора.

    - per_class (UnsafeBench): до `n` примеров на КАЖДУЮ ячейку (категория×label).
    - binary: 50/50 unsafe/safe со стратификацией unsafe по категориям.
    - прочее: первые n из перемешанного порядка.
    """
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, spec.config, split=spec.split, token=_hf_token(spec.access, token))
    features = getattr(ds, "features", None)
    labeler = _build_labeler(spec, features)
    rng = random.Random(seed)

    order = list(range(ds.num_rows))
    rng.shuffle(order)

    if spec.per_class:
        return _collect_per_class(ds, spec, order, n, progress)

    if spec.kind == "binary":
        idx_iter = _balanced_order(ds, spec, labeler, order, n)
    else:
        idx_iter = order[:n]

    items: list[tuple[Image.Image, Label]] = []
    for i in idx_iter:
        try:
            ex = ds[i]
            img = _extract_image(ex, spec.image_fields)
        except Exception:
            continue  # битый пример (напр. WEBP без libwebp) — пропускаем
        if img is None:
            continue
        label = labeler(ex)
        if label is None:
            continue
        items.append((img, label))
        if progress is not None:
            progress(1)
    return items


def _collect_per_class(ds, spec, order, n, progress):
    """Собрать до `n` ПРИГОДНЫХ картинок на каждый класс (категория × safe/unsafe).

    Класс определяем дёшево по колонкам (без декода), декодируем только пока
    класс не насыщен. Битые/неподдерживаемые картинки (WEBP без libwebp) просто
    пропускаются — класс добирается из следующих примеров.
    """
    labeler = _build_labeler(spec, getattr(ds, "features", None))
    label_col, cols = _label_and_cat_columns(ds, spec)
    lab_arr = cols.get(label_col)
    cat_arr = cols.get("category")
    positives = {v.lower() for v in spec.positive_values}

    collected: dict[tuple[str, str | None], int] = {}
    items: list[tuple[Image.Image, Label]] = []
    for i in order:
        is_unsafe = str(lab_arr[i]).strip().lower() in positives if lab_arr is not None else False
        tax = UNSAFEBENCH_TO_TAXONOMY.get(str(cat_arr[i])) if cat_arr is not None else None
        key = ("U" if is_unsafe else "S", tax)
        if collected.get(key, 0) >= n:
            continue  # класс уже добран — не тратим время на декод
        try:
            ex = ds[i]
            img = _extract_image(ex, spec.image_fields)
        except Exception:
            continue
        if img is None:
            continue
        label = labeler(ex)
        if label is None:
            continue
        items.append((img, label))
        collected[key] = collected.get(key, 0) + 1
        if progress is not None:
            progress(1)
    return items


def _label_and_cat_columns(ds, spec):
    """Дешёво вернуть (label_col_name, cols-dict) без декода картинок."""
    label_col = next((f for f in spec.label_fields if features_has(ds, f)), None)
    cols = {label_col: ds[label_col]} if label_col else {}
    if features_has(ds, "category"):
        cols["category"] = ds["category"]
    return label_col, cols


def _balanced_order(ds, spec, labeler, order, n):
    """Очередь индексов: 50/50 unsafe/safe со стратификацией unsafe по категориям."""
    label_col, cols = _label_and_cat_columns(ds, spec)
    pos_by_cat: dict[str | None, list[int]] = {}
    neg: list[int] = []
    for i in order:
        ex = {k: v[i] for k, v in cols.items()}
        label = labeler(ex)
        if label is None or label.true_unsafe is None:
            continue
        if label.true_unsafe:
            pos_by_cat.setdefault(label.true_category, []).append(i)
        else:
            neg.append(i)

    half = max(1, n // 2)
    cat_lists = [lst for lst in pos_by_cat.values() if lst]
    pos: list[int] = []
    ci = 0
    while cat_lists and len(pos) < half:
        lst = cat_lists[ci % len(cat_lists)]
        if lst:
            pos.append(lst.pop())
        if not lst:
            cat_lists.remove(lst)
            ci -= 1 if cat_lists else 0
        ci += 1
    chosen = pos[:half] + neg[: n - len(pos[:half])]
    random.Random(0).shuffle(chosen)
    return chosen


def features_has(ds, field_name: str) -> bool:
    feats = getattr(ds, "features", None)
    return bool(feats) and field_name in feats


def sample_dataset(spec, n, seed=42, time_budget=1800, token=None, progress=None) -> SampleResult:
    """Загрузить до `n` картинок из `spec`. Любую недоступность → мягкий skip.

    `progress`: опциональный callback(int) — инкремент глобального прогресс-бара
    при загрузке (на каждую успешно взятую картинку).
    """
    if spec.access == "kaggle":
        return SampleResult(spec, skipped_reason="требует Kaggle API (KAGGLE_USERNAME/KEY) — пропущен")
    if spec.access == "request":
        return SampleResult(spec, skipped_reason="доступ только по запросу к авторам — пропущен")
    if spec.access == "hf_token" and not token:
        return SampleResult(spec, skipped_reason="нет HF_TOKEN для gated-датасета — пропущен")

    t0 = time.time()
    try:
        if spec.streaming:
            items = _sample_streaming(spec, n, seed, time_budget, token, progress)
        else:
            items = _sample_indexed(spec, n, seed, token, progress)
    except Exception as exc:
        return SampleResult(spec, skipped_reason=f"{type(exc).__name__}: {str(exc)[:120]}", load_seconds=time.time() - t0)

    res = SampleResult(spec, items=items, load_seconds=time.time() - t0)
    if not items:
        res.skipped_reason = "0 пригодных картинок (пустой/несовместимый набор)"
    return res


def select_specs(keys: list[str] | None) -> list[DatasetSpec]:
    """Отфильтровать реестр по списку ключей ИЛИ категорий. None → весь реестр."""
    if not keys:
        return list(REGISTRY)
    wanted = {k.strip() for k in keys if k.strip()}
    out = [s for s in REGISTRY if s.key in wanted or s.category in wanted]
    return out or list(REGISTRY)
