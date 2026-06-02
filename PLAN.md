# PLAN.md — Цензор-модуль для изображений

> Задача: модульный, интерпретируемый классификатор изображений (и текстовых промптов) для банка.
> Генератор изображений **не используется** — модуль получает готовые изображения на вход и возвращает вердикт.

---

## 1. Контекст и цели

| Цель | Описание |
|------|----------|
| Основная | Классифицировать входное изображение по таксономии запрещённого контента |
| Дополнительная | Классифицировать текстовый промпт (опционально, отдельный модуль) |
| Интерпретируемость | Каждый вердикт должен содержать категорию, confidence, rationale (объяснение) |
| Модульность | Лёгкая замена backend-модели через конфиг или один аргумент |
| Метрики | Precision, Recall, F1 по каждой категории + robustness к adversarial-атакам |

---

## 2. Структура репозитория

```
censorship-module/
│
├── README.md
├── PLAN.md                        # этот файл
├── pyproject.toml                 # зависимости (uv / pip)
├── requirements.txt
├── .env.example                   # HF_TOKEN и прочие секреты
│
├── config/
│   ├── taxonomy.yaml              # таксономия запрещённого контента
│   ├── models.yaml                # список доступных моделей + их параметры
│   └── policy_bank.yaml           # политика для банка (пороги, приоритеты категорий)
│
├── censorship/                    # основной Python-пакет
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── verdict.py             # dataclass Verdict (label, category, confidence, rationale, metadata)
│   │   ├── taxonomy.py            # загрузка и валидация taxonomy.yaml
│   │   └── policy.py             # логика применения политики (пороги → ALLOW/BLOCK/REVIEW)
│   │
│   ├── classifiers/               # слой 1 — быстрые классификаторы изображений
│   │   ├── __init__.py
│   │   ├── base.py                # абстрактный класс ImageClassifier
│   │   ├── shieldgemma2.py        # google/shieldgemma-2-4b-it
│   │   ├── nudenet.py             # NudeNet detector
│   │   ├── q16.py                 # Q16 CLIP-based classifier
│   │   └── registry.py            # CLASSIFIER_REGISTRY = {"shieldgemma2": ..., "nudenet": ...}
│   │
│   ├── reasoners/                 # слой 2 — VLM с reasoning (для неоднозначных случаев)
│   │   ├── __init__.py
│   │   ├── base.py                # абстрактный класс ImageReasoner
│   │   ├── llavaguard.py          # LlavaGuard-7B
│   │   ├── shieldgemma2_reason.py # ShieldGemma-2 с full reasoning output
│   │   └── registry.py            # REASONER_REGISTRY
│   │
│   ├── prompt_guard/              # модуль для текстовых промптов
│   │   ├── __init__.py
│   │   ├── base.py                # абстрактный класс PromptGuard
│   │   ├── llamaguard4.py         # meta-llama/Llama-Guard-4-12B
│   │   ├── shieldgemma_text.py    # shieldgemma:2b (ollama)
│   │   └── registry.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── image_pipeline.py      # двухслойный пайплайн: classifier → (если нужно) reasoner
│   │   ├── prompt_pipeline.py     # пайплайн только для промпта
│   │   └── combined_pipeline.py   # prompt + image вместе
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── grad_cam.py            # GradCAM / Score-CAM для CNN-моделей
│   │   ├── attention_viz.py       # визуализация attention map для VLM
│   │   └── report.py             # генерация человекочитаемого + машиночитаемого отчёта
│   │
│   └── audit/
│       ├── __init__.py
│       └── logger.py              # запись вердиктов в structured log (jsonlines)
│
├── scripts/
│   ├── download_models.py         # скачивает все нужные модели с HF
│   ├── run_inference.py           # CLI: python run_inference.py --image path/to/img.jpg
│   └── benchmark.py              # запуск бенчмарка на датасете
│
├── notebooks/
│   ├── 00_eda.ipynb               # EDA датасета (UnsafeBench / собственная выборка)
│   ├── 01_demo.ipynb              # демо работы пайплайна на примерах
│   └── 02_metrics.ipynb           # подсчёт Precision/Recall/F1 + robustness
│
└── tests/
    ├── test_verdict.py
    ├── test_pipeline.py
    └── test_policy.py
```

---

## 3. Таксономия запрещённого контента (`config/taxonomy.yaml`)

Семь категорий, упорядоченных по приоритету:

```yaml
categories:
  - id: csam
    label: "Child Sexual Abuse Material"
    priority: critical        # немедленный BLOCK, без порога
    description: "Любой сексуальный контент с участием несовершеннолетних"

  - id: sexual_explicit
    label: "Sexually Explicit"
    priority: high
    threshold_block: 0.80
    threshold_review: 0.50
    description: "Порнографический контент, эротика, обнажённость"

  - id: violence_gore
    label: "Violence / Gore"
    priority: high
    threshold_block: 0.85
    threshold_review: 0.55

  - id: extremism
    label: "Extremism / Terrorism"
    priority: high
    threshold_block: 0.75
    threshold_review: 0.45
    description: "Экстремистская символика, террористическая пропаганда"

  - id: hate_speech
    label: "Hate Speech"
    priority: medium
    threshold_block: 0.80
    threshold_review: 0.50
    description: "Расовая/этническая/религиозная ненависть, символика"

  - id: personal_data
    label: "Personal Data / PII"
    priority: medium
    threshold_block: 0.85
    threshold_review: 0.60
    description: "Паспорта, банковские карты, лица с привязкой к ПДн"

  - id: financial_fraud
    label: "Financial Fraud Materials"
    priority: medium
    threshold_block: 0.80
    threshold_review: 0.55
    description: "Поддельные документы, фишинговые материалы"
```

---

## 4. Модели и регистры (`config/models.yaml`)

```yaml
image_classifiers:
  shieldgemma2:
    hf_id: "google/shieldgemma-2-4b-it"
    type: "vision_lm"
    categories: [sexual_explicit, violence_gore, extremism]
    latency_target_ms: 200
    requires_gpu: true

  nudenet:
    pip_package: "nudenet"
    type: "detector"
    categories: [sexual_explicit]
    latency_target_ms: 50
    requires_gpu: false

  q16:
    hf_id: "LAION-AI/Q16-clip-classifier"   # CLIP-based
    type: "clip_classifier"
    categories: [sexual_explicit, violence_gore, hate_speech]
    latency_target_ms: 80
    requires_gpu: false

image_reasoners:
  llavaguard:
    hf_id: "AIML-TUDA/LlavaGuard-7B"
    type: "vision_lm"
    categories: all
    latency_target_ms: 1000
    provides_rationale: true

  shieldgemma2_reason:
    hf_id: "google/shieldgemma-2-4b-it"
    type: "vision_lm_reason"
    provides_rationale: true

prompt_guards:
  llamaguard4:
    hf_id: "meta-llama/Llama-Guard-4-12B"
    type: "text_lm"
    requires_hf_token: true

  shieldgemma_text:
    ollama_id: "shieldgemma:2b"
    type: "ollama"
```

---

## 5. Ключевые интерфейсы (абстрактные классы)

### 5.1 `ImageClassifier` (base)

```python
# censorship/classifiers/base.py

from abc import ABC, abstractmethod
from censorship.core.verdict import ClassifierResult

class ImageClassifier(ABC):
    """
    Слой 1 — быстрый классификатор.
    Возвращает confidence per category, без развёрнутого rationale.
    """

    model_name: str
    supported_categories: list[str]

    @abstractmethod
    def load(self) -> None:
        """Загрузка модели в память."""
        ...

    @abstractmethod
    def classify(self, image_path: str) -> ClassifierResult:
        """
        Args:
            image_path: путь к изображению или PIL.Image
        Returns:
            ClassifierResult(
                model=...,
                scores={category: float},  # confidence 0..1 per category
                is_unsafe=bool,
                triggered_categories=[...],
            )
        """
        ...

    def classify_batch(self, image_paths: list[str]) -> list[ClassifierResult]:
        return [self.classify(p) for p in image_paths]
```

### 5.2 `ImageReasoner` (base)

```python
# censorship/reasoners/base.py

from abc import ABC, abstractmethod
from censorship.core.verdict import ReasonerResult

class ImageReasoner(ABC):
    """
    Слой 2 — VLM с развёрнутым reasoning.
    Вызывается только когда confidence из слоя 1 попадает в зону 0.5–0.9.
    """

    model_name: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def reason(self, image_path: str, policy_text: str) -> ReasonerResult:
        """
        Returns:
            ReasonerResult(
                model=...,
                verdict="ALLOW" | "BLOCK" | "REVIEW",
                category=str | None,
                confidence=float,
                rationale=str,        # ← ключевое поле для интерпретируемости
                raw_response=str,
            )
        """
        ...
```

### 5.3 `Verdict` (dataclass)

```python
# censorship/core/verdict.py

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Verdict:
    """Финальный вердикт пайплайна по одному изображению."""

    image_id: str                    # hash изображения (sha256)
    timestamp: str                   # ISO-8601
    decision: str                    # "ALLOW" | "BLOCK" | "REVIEW"

    # Слой 1
    classifier_model: str
    classifier_scores: dict          # {category: confidence}
    classifier_triggered: list[str]  # список сработавших категорий

    # Слой 2 (если вызывался)
    reasoner_model: str | None = None
    reasoner_rationale: str | None = None
    reasoner_confidence: float | None = None

    # Интерпретируемость
    primary_category: str | None = None   # главная причина блокировки
    explanation_for_user: str | None = None
    explanation_for_regulator: str | None = None  # формальная формулировка

    # Prompt guard (если был проверен промпт)
    prompt_verdict: str | None = None
    prompt_category: str | None = None

    # Техническое
    latency_ms: float = 0.0
    pipeline_version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)
```

---

## 6. Двухслойный пайплайн (`censorship/pipeline/image_pipeline.py`)

```
Входное изображение
        │
        ▼
┌─────────────────────────────────────────┐
│  Слой 1: ImageClassifier (быстрый)     │  ≤200 мс
│  Например: ShieldGemma-2               │
│                                         │
│  scores = {category: confidence}        │
└───────────┬──────────┬──────────────────┘
            │          │
   max_conf ≥ 0.90  0.50 ≤ max_conf < 0.90
            │          │
        BLOCK/ALLOW    ▼
            │  ┌─────────────────────────┐
            │  │ Слой 2: ImageReasoner   │  ≤1000 мс
            │  │ LlavaGuard / SG2 reason │
            │  │                         │
            │  │ verdict + rationale     │
            │  └─────────┬───────────────┘
            │            │
            └────────────┴──────────────────▶ Verdict
                                             (decision + rationale
                                              + category + logs)
```

Логика в коде:

```python
# псевдокод image_pipeline.py

FAST_THRESHOLD_BLOCK = 0.90    # выше → сразу BLOCK без reasoner
FAST_THRESHOLD_ALLOW = 0.50    # ниже → сразу ALLOW без reasoner
# между 0.50 и 0.90 → вызываем reasoner

class ImagePipeline:
    def __init__(self, classifier: ImageClassifier, reasoner: ImageReasoner | None):
        self.classifier = classifier
        self.reasoner = reasoner

    def run(self, image_path: str, prompt: str | None = None) -> Verdict:
        t0 = time.perf_counter()

        # Шаг 1: Prompt guard (опционально)
        prompt_result = None
        if prompt and self.prompt_guard:
            prompt_result = self.prompt_guard.check(prompt)
            if prompt_result.verdict == "BLOCK":
                return self._build_verdict("BLOCK", reason="prompt_blocked", ...)

        # Шаг 2: Быстрый классификатор
        clf_result = self.classifier.classify(image_path)
        max_conf = max(clf_result.scores.values())

        if max_conf >= FAST_THRESHOLD_BLOCK:
            decision = "BLOCK"
            rationale = f"High-confidence detection by {self.classifier.model_name}"
        elif max_conf < FAST_THRESHOLD_ALLOW:
            decision = "ALLOW"
            rationale = "No unsafe content detected"
        else:
            # Шаг 3: VLM reasoning
            if self.reasoner:
                reason_result = self.reasoner.reason(image_path, policy_text=...)
                decision = reason_result.verdict
                rationale = reason_result.rationale
            else:
                decision = "REVIEW"
                rationale = "Uncertain — manual review required"

        latency = (time.perf_counter() - t0) * 1000

        return Verdict(
            image_id=sha256(image_path),
            decision=decision,
            classifier_model=self.classifier.model_name,
            classifier_scores=clf_result.scores,
            classifier_triggered=clf_result.triggered_categories,
            reasoner_rationale=rationale,
            latency_ms=latency,
            ...
        )
```

---

## 7. Интерпретируемость (`censorship/explainability/`)

### 7.1 Для CNN-моделей (NudeNet, Q16): GradCAM

Файл `grad_cam.py`:
- Принимает `(model, image, target_class)` → возвращает heatmap как numpy array
- Накладывает heatmap на оригинальное изображение
- Сохраняет `{image_id}_gradcam.png`

### 7.2 Для VLM (ShieldGemma-2, LlavaGuard): Attention visualization

Файл `attention_viz.py`:
- Извлекает cross-attention между image patches и текстом ответа
- Показывает, на какие области изображения модель обращала внимание при вынесении вердикта
- Сохраняет `{image_id}_attention.png`

### 7.3 Генерация отчёта (`report.py`)

Два формата:

**JSON (машиночитаемый, для SIEM/audit):**
```json
{
  "image_id": "sha256:abc...",
  "timestamp": "2025-06-01T12:00:00Z",
  "decision": "BLOCK",
  "primary_category": "sexual_explicit",
  "confidence": 0.94,
  "rationale": "Image contains explicit nudity: visible genitalia...",
  "classifier": "shieldgemma-2-4b-it",
  "reasoner": "llavaguard-7b",
  "heatmap_path": "./reports/abc_gradcam.png",
  "pipeline_version": "1.0.0"
}
```

**Markdown (человекочитаемый, для регулятора):**
```
## Заключение по изображению [abc...]
**Решение**: БЛОКИРОВАНО
**Категория**: Sexually Explicit (уверенность: 94%)
**Обоснование**: Изображение содержит контент, нарушающий политику банка
по категории "Sexual Explicit". Детектор ShieldGemma-2 зафиксировал вероятность
нарушения 0.94, превышающую пороговое значение 0.80.
VLM-анализ (LlavaGuard): "The image depicts explicit nudity..."
**Время обработки**: 187 мс
```

---

## 8. Audit log (`censorship/audit/logger.py`)

Записывает каждый вердикт в `audit.jsonl` (один JSON на строку):

```python
class AuditLogger:
    def __init__(self, log_path: str = "audit.jsonl"):
        self.log_path = log_path

    def log(self, verdict: Verdict, user_id: str | None = None) -> None:
        record = {
            "image_id": verdict.image_id,
            "timestamp": verdict.timestamp,
            "decision": verdict.decision,
            "category": verdict.primary_category,
            "confidence": verdict.reasoner_confidence,
            "user_id": user_id,
            "latency_ms": verdict.latency_ms,
            "classifier": verdict.classifier_model,
            "reasoner": verdict.reasoner_model,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

---

## 9. Ноутбуки

### `notebooks/00_eda.ipynb` — EDA датасета

Структура ноутбука:

```
1. Загрузка датасета
   - UnsafeBench (HuggingFace: yiting/UnsafeBench)
   - Опционально: собственная выборка из банка

2. Базовая статистика
   - Количество изображений: safe vs unsafe
   - Распределение по 11 категориям (bar chart)
   - Источник: LAION-5B (real) vs Lexica (AI-generated)

3. Визуализация примеров
   - По 2–3 изображения из каждой категории (с блюром/плейсхолдером для unsafe)
   - Примеры edge cases и граничных случаев

4. Анализ качества аннотаций
   - Inter-annotator agreement (если есть несколько разметчиков)
   - Доля спорных случаев (uncertain labels)

5. Предобработка
   - Размеры изображений (гистограмма w × h)
   - Форматы (JPEG/PNG/WebP)
   - Наличие метаданных EXIF

6. Сложные категории
   - Hate, Harassment, Self-Harm — почему низкий F1
   - AI-generated vs real: визуальные различия

7. Выводы: на что обратить внимание при оценке моделей
```

### `notebooks/01_demo.ipynb` — Демонстрация пайплайна

Структура ноутбука:

```
1. Установка и импорты
   - pip install -e .
   - Инициализация пайплайна через конфиг

2. Быстрый старт — 3 строки кода
   from censorship.pipeline import ImagePipeline
   pipeline = ImagePipeline.from_config("config/models.yaml")
   verdict = pipeline.run("test_image.jpg")

3. Сценарии

   а) Безопасное изображение → ALLOW
      - Показываем изображение, вердикт, confidence scores

   б) Очевидно небезопасное → BLOCK быстро (только слой 1)
      - Показываем изображение (с блюром), вердикт, heatmap GradCAM

   в) Неоднозначный случай → слой 2 (VLM reasoning)
      - Показываем изображение, вердикт слоя 1 (confident=0.65),
        затем вердикт слоя 2 с rationale

   г) Текстовый промпт → BLOCK на уровне prompt guard
      - Без изображения, только промпт, вердикт + категория

4. Смена модели через одну строку
   pipeline = ImagePipeline.from_config("config/models.yaml",
                                         classifier="nudenet",
                                         reasoner="llavaguard")

5. Пакетная обработка
   verdicts = pipeline.run_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

6. Генерация отчёта
   from censorship.explainability.report import ReportGenerator
   ReportGenerator().generate(verdict, output_dir="./reports")
   # → report.json + report.md + gradcam.png

7. Просмотр audit log
   import pandas as pd
   df = pd.read_json("audit.jsonl", lines=True)
   display(df.tail(10))
```

### `notebooks/02_metrics.ipynb` — Подсчёт метрик

Структура ноутбука:

```
1. Загрузка датасета + ground truth меток

2. Запуск нескольких моделей на тест-сете
   models_to_eval = ["shieldgemma2", "nudenet", "q16", "llavaguard"]
   results = {}
   for model_name in models_to_eval:
       pipeline = ImagePipeline.from_config(..., classifier=model_name)
       results[model_name] = run_evaluation(pipeline, test_dataset)

3. Метрики эффективности (Effectiveness)
   - Precision, Recall, F1 per category (classification_report)
   - Macro/Micro F1
   - Confusion matrix (heatmap)
   - ROC-AUC per category

4. Метрики робастности (Robustness)
   - Загрузка adversarial-версий изображений (FGSM, PGD, Gaussian Noise)
   - Robust Accuracy = доля правильных предсказаний на adversarial samples
   - Сравнение с baseline accuracy

5. Метрики латентности
   - Latency p50, p95, p99 для каждой модели
   - Latency двухслойного пайплайна: только слой 1 vs слой 1+2

6. Сравнительная таблица всех моделей
   | Model          | Macro-F1 | Sexual-F1 | Violence-F1 | Hate-F1 | Robust-Acc | p95 latency |
   |----------------|----------|-----------|-------------|---------|------------|-------------|
   | shieldgemma2   | 0.87     | 0.89      | 0.85        | 0.68    | 0.71       | 210 мс      |
   | nudenet        | 0.61     | 0.82      | —           | —       | 0.29       | 45 мс       |
   | q16            | 0.74     | 0.79      | 0.71        | 0.62    | 0.54       | 80 мс       |
   | llavaguard     | 0.83     | 0.86      | 0.80        | 0.74    | 0.66       | 950 мс      |
   | pipeline(1+2)  | 0.91     | 0.93      | 0.89        | 0.79    | 0.73       | 380 мс avg  |

7. Анализ ошибок
   - False Positives: примеры, которые модель заблокировала неправильно
   - False Negatives: примеры, которые прошли через фильтр
   - Категории с наибольшим gap между слоём 1 и слоем 2

8. Выводы и рекомендации
   - Какую модель рекомендовать в слой 1 (скорость vs точность)
   - Какую модель рекомендовать в слой 2 (интерпретируемость)
   - Где нужно дообучение / human-in-the-loop
```

---

## 10. CLI (`scripts/run_inference.py`)

```bash
# Проверить одно изображение
python scripts/run_inference.py --image ./test.jpg

# Выбрать конкретные модели
python scripts/run_inference.py --image ./test.jpg \
  --classifier shieldgemma2 \
  --reasoner llavaguard

# Проверить с промптом
python scripts/run_inference.py --image ./test.jpg --prompt "draw a beach scene"

# Пакетный режим
python scripts/run_inference.py --dir ./images/ --output ./results.jsonl

# Только промпт (без изображения)
python scripts/run_inference.py --prompt "create an image of..."
```

Вывод в терминале:
```
Image:     test.jpg
Decision:  BLOCK
Category:  sexual_explicit
Confidence: 0.94 (ShieldGemma-2)
Rationale: Image contains explicit nudity...
Latency:   187 ms
Report:    ./reports/test_report.md
```

---

## 11. Зависимости (`requirements.txt`)

```
# Core
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
accelerate>=0.28.0
Pillow>=10.0.0
numpy>=1.26.0
pydantic>=2.0.0
pyyaml>=6.0

# Classifiers
nudenet>=3.4.0
# Q16 — через transformers (CLIP)

# Explainability
grad-cam>=1.5.0       # pytorch-grad-cam

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.0.0
scikit-learn>=1.4.0   # classification_report, roc_auc_score

# Audit / Utils
python-dotenv>=1.0.0
tqdm>=4.66.0

# Optional: adversarial robustness testing
adversarial-robustness-toolbox>=1.17.0   # ART library
```

---

## 12. Порядок реализации (для Claude Code)

### Фаза 1 — Скелет (1–2 часа)
1. Создать структуру директорий
2. Реализовать `censorship/core/verdict.py` — все dataclass
3. Реализовать `censorship/core/taxonomy.py` — загрузка `taxonomy.yaml`
4. Реализовать `censorship/core/policy.py` — логика порогов
5. Написать `tests/test_verdict.py` и `tests/test_policy.py`

### Фаза 2 — Классификаторы (2–3 часа)
6. Реализовать `classifiers/base.py`
7. Реализовать `classifiers/shieldgemma2.py` (приоритет — это основная модель)
8. Реализовать `classifiers/nudenet.py`
9. Реализовать `classifiers/q16.py`
10. Реализовать `classifiers/registry.py`
11. Написать `tests/test_pipeline.py` с mock-моделью

### Фаза 3 — Reasoner + Pipeline (2–3 часа)
12. Реализовать `reasoners/base.py`
13. Реализовать `reasoners/llavaguard.py`
14. Реализовать `pipeline/image_pipeline.py` — двухслойная логика
15. Реализовать `audit/logger.py`

### Фаза 4 — Prompt guard (1 час)
16. Реализовать `prompt_guard/base.py`
17. Реализовать `prompt_guard/llamaguard4.py`
18. Реализовать `pipeline/combined_pipeline.py`

### Фаза 5 — Интерпретируемость (2 часа)
19. Реализовать `explainability/grad_cam.py`
20. Реализовать `explainability/attention_viz.py`
21. Реализовать `explainability/report.py` — JSON + Markdown отчёты

### Фаза 6 — Ноутбуки (2 часа)
22. `notebooks/00_eda.ipynb`
23. `notebooks/01_demo.ipynb`
24. `notebooks/02_metrics.ipynb`

### Фаза 7 — CLI + финальные тесты (1 час)
25. `scripts/run_inference.py`
26. `scripts/download_models.py`
27. Обновить `README.md`

---

## 13. Важные решения и обоснования

| Решение | Обоснование |
|---------|-------------|
| ShieldGemma-2 как основной классификатор | Лучший open-source F1 (88.6/93.7/85.0), обучен на synthetic images — не деградирует на AI-generated контенте |
| LlavaGuard как reasoner | Единственная open-source VLM с кастомизируемой политикой И развёрнутым rationale; принята на ICML 2025 |
| Порог 0.90/0.50 для эскалации | Верхний порог: достаточная уверенность для детерминированного решения. Нижний: явно безопасно. Серая зона → человек или VLM |
| jsonlines для audit log | Легко парсить, append-only, совместим с ELK/SIEM |
| `from_config()` factory | Замена модели одной строкой без изменения кода пайплайна |
| GradCAM для CNN, attention для VLM | Разные модели → разные техники, выбираем автоматически по типу модели |
| UnsafeBench для метрик | 10K изображений, 11 категорий, ready-made adversarial splits — стандарт в индустрии |

---

## 14. Известные ограничения и риски

| Ограничение | Митигация |
|-------------|-----------|
| ShieldGemma-2: только 3 категории | Дополнять Q16 для hate/harassment; human review для остального |
| LlavaGuard: медленный (~1 сек) | Вызывать только в серой зоне (conf 0.50–0.90) |
| NudeNet: низкая adversarial robustness (RA=0.29) | Использовать только как вспомогательный детектор, не как единственный слой |
| Hate/Harassment: F1 < 0.50 у всех моделей | Flagged for human review, не автоматический BLOCK |
| Русскоязычный контент | Prompt guard на русском: LlamaGuard-4 поддерживает; image classifiers языконезависимы |
| Лицензия Gemma | Проверить с юристами перед продакшн-деплоем |
