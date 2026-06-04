# План развития Censor Module

Документ описывает, что уже сделано в рефакторе пайплайна и что осталось.
Текущую архитектуру см. в [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ✅ Сделано (рефактор финальных стадий пайплайна)

1. **Текстовый гард — реальный, а не заглушка** (`adapters/text_classifier.py`,
   сигнал `text_guard_heuristic`). ML-модель по умолчанию
   `cointegrated/rubert-tiny-toxicity` (`CENSOR_TEXT_MODEL_ID`) + лексикон ru/en
   (`taxonomy.TEXT_LEXICONS`), слитые по максимуму. Применяется и к промпту, и к
   OCR-тексту (`ocr_text_guard_heuristic`).
2. **Калибровка zero-shot CLIP** (`calibration.py`). Сырой softmax больше не
   уходит в решение: оценки калибруются относительно нескольких safe-якорей, так
   что безобидная картинка коллапсирует к ~0. Порог — `CENSOR_CALIBRATION_FLOOR`.
3. **Принципиальный fusion** (`fusion.py`): взвешенный noisy-OR с прозрачным
   вкладом каждого сенсора. Убраны бессмысленный `evidence_count` (был константой
   из-за softmax-шума CLIP) и магические `±0.10/−0.05`.
4. **Судья переосмыслен** (`adapters/policy_judge.py`): fusion как основа +
   ShieldGemma как эскалационный слой на пограничных случаях. Честный `skipped`
   вместо фейкового `fallback_from`.
5. **DecisionEngine** работает на откалиброванных fused-оценках; soft-блок требует
   согласия ≥2 сенсоров или подтверждения ShieldGemma.
6. **Тесты**: калибровка, fusion, текст-классификатор, решение на fusion.

---

## 🔜 Следующие шаги (по приоритету)

### 1. Подключить инференс ShieldGemma
Точка интеграции готова (`ShieldGemmaJudge.moderate` получает image + prompt +
evidence). Нужно вшить загрузку модели и парсинг ответа в оценки по категориям.
Вызывается только на эскалации (дёшево). Веса: `CENSOR_POLICY_JUDGE_MODEL_ID`.

### 2. Расширить покрытие текстового слоя
ML-модель по умолчанию (`rubert-tiny-toxicity`) сильна в harassment/threat, но не
различает sexual / политику / мед-дезинформацию — там работает только лексикон
(обходится перефразировкой). Варианты усиления: добавить EN-toxicity для
англоязычных промптов, либо перейти на модель уровня OpenAI-moderation
(`KoalaAI/Text-Moderation`) или LLM-guard (Llama-Guard / ShieldGemma) с прямым
покрытием категорий. Маппинг меток — в `text_classifier.ML_LABEL_MARKERS`.

### 3. Калибровка порогов на датасете
Прогнать UnsafeBench (`evaluation.py` уже готов: `ImageClassifierRunner`,
`ClassifierResult.unsafe_score`) и подобрать `calibration_floor`,
`review_threshold`, `block_threshold` и веса сенсоров (`fusion.DEFAULT_SENSOR_WEIGHTS`)
по ROC/PR-кривым.

### 4. Дифференциация политик по scenario/stage
Сейчас `scenario` на решение не влияет, `stage` — только на эскалацию. Ввести
профили политик (например, output строже input; editing строже stylization).

### 5. Эксплуатация
- Асинхронный инференс / вынос моделей в воркеры (сейчас ручка синхронная).
- Структурное логирование вердиктов и сигналов для аудита инцидентов.
- Метрики (доля block/review/allow, латентность по сенсорам).

### 6. Расширение специалистов
`explicit_content_detector` покрывает только `sexual`. Добавить специалистов под
другие hard-категории (насилие/gore, экстремистская символика) — архитектура
fusion примет их как новые сенсоры с собственными весами.
