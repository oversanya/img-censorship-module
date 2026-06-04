# Датасеты для бенчмарка цензор-модуля

> **Статус верификации:** каждый датасет проверен на реальное существование по прямому URL на HuggingFace.  
> **Доступ:** датасеты без специальных пометок загружаются через `load_dataset("id")`;  
> датасеты с 🔒 требуют принятия условий в браузере + `HF_TOKEN` в переменной окружения;  
> датасеты с ⚠️ требуют ручного скачивания по отдельной инструкции.

---

## Содержание

1. [Класс: SEXUAL](#1-класс-sexual)
2. [Класс: VIOLENCE / GORE](#2-класс-violence--gore)
3. [Класс: HATEFUL / EXTREMIST (мемы, в т.ч. RU)](#3-класс-hateful--extremist)
4. [Класс: DISTURBING / SELF-HARM](#4-класс-disturbing--self-harm)
5. [Класс: ADVERSARIAL (обход цензуры)](#5-класс-adversarial)
6. [SAFE-негативы (false positive тест)](#6-safe-негативы)
7. [Рекомендованный состав тестовой выборки](#7-рекомендованный-состав-тестовой-выборки)
8. [Шаблон загрузки](#8-шаблон-загрузки)

---

## 1. Класс: SEXUAL

### 1.1 `yesidobyte/nsfw1024`
- **URL:** https://huggingface.co/datasets/yesidobyte/nsfw1024
- **Доступ:** публичный, без токена
- **Размер:** ~10 000 изображений, 1024×1024 px
- **Описание:** Высокое разрешение. Сексуально явный контент, преимущественно реалистичные фото. Хорошая вариативность поз и типов.
- **Загрузка:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("yesidobyte/nsfw1024")
  ```

### 1.2 `wallstoneai/civitai-top-nsfw-images-with-metadata`
- **URL:** https://huggingface.co/datasets/wallstoneai/civitai-top-nsfw-images-with-metadata
- **Доступ:** публичный, без токена
- **Размер:** крупный (~100k+), с метаданными (модель, промпт, LORA)
- **Описание:** AI-генерация (Stable Diffusion / Flux) с CivitAI. Важен для проверки детектора на синтетический контент. Метаданные позволяют стратифицировать по генератору.
- **Загрузка:**
  ```python
  ds = load_dataset("wallstoneai/civitai-top-nsfw-images-with-metadata")
  ```

### 1.3 `deepghs/nsfw_detect` 🔒
- **URL:** https://huggingface.co/datasets/deepghs/nsfw_detect
- **Доступ:** требует принятия условий + `HF_TOKEN`
- **Размер:** 10 000–100 000 изображений, 1.8 GB
- **Описание:** Мультиклассовый датасет с 5 категориями: `drawing`, `hentai`, `neutral`, `porn`, `sexy`. Охватывает как фото, так и аниме/манга-стиль. Лицензия MIT. Совместим с форматом GantMan/nsfw_model.
- **Загрузка:**
  ```python
  import os
  ds = load_dataset("deepghs/nsfw_detect", token=os.environ["HF_TOKEN"])
  # Или скачать zip напрямую:
  # wget https://huggingface.co/datasets/deepghs/nsfw_detect/resolve/main/nsfw_dataset_v1.zip
  ```

### 1.4 `QuixiAI/SexDrugsAndRockAndRoll`
- **URL:** https://huggingface.co/datasets/QuixiAI/SexDrugsAndRockAndRoll
- **Доступ:** публичный, без токена
- **Размер:** средний (~5–10k), мультикатегория
- **Описание:** Мультимодальный датасет с детальными метаданными: adult content, drug imagery, violence, nudity. Предназначен для обучения VLM. Хорош тем, что покрывает несколько классов одновременно (sexual + drugs).
- **Загрузка:**
  ```python
  ds = load_dataset("QuixiAI/SexDrugsAndRockAndRoll")
  ```

### 1.5 `amaye15/NSFW`
- **URL:** https://huggingface.co/datasets/amaye15/NSFW
- **Доступ:** публичный, без токена
- **Размер:** средний
- **Описание:** Бинарный датасет (safe/nsfw), реальные фото. Хорош как дополнительный пул для увеличения разнообразия.
- **Загрузка:**
  ```python
  ds = load_dataset("amaye15/NSFW")
  ```

---

## 2. Класс: VIOLENCE / GORE

### 2.1 `NeuralShell/Gore-Blood-Dataset-v1.0` 🔒
- **URL:** https://huggingface.co/datasets/NeuralShell/Gore-Blood-Dataset-v1.0
- **Доступ:** marked Not-For-All-Audiences, требует принятия условий + `HF_TOKEN`
- **Размер:** < 1K изображений (небольшой), задачи: image-to-image, classification, segmentation
- **Описание:** Специализированный датасет с кровью и жестокостью для обучения/тестирования детекторов. Лицензия MIT. Небольшой объём — использовать как seed для разметки, не как основной источник.
- **Загрузка:**
  ```python
  ds = load_dataset("NeuralShell/Gore-Blood-Dataset-v1.0", token=os.environ["HF_TOKEN"])
  ```

### 2.2 `jherng/xd-violence`
- **URL:** https://huggingface.co/datasets/jherng/xd-violence
- **Доступ:** публичный, без токена
- **Размер:** крупный (~4k видео-клипов, 13 классов)
- **Описание:** Мультимодальный датасет насилия в видео: abuse, car accident, explosion, fighting, riot, shooting, and more. Для бенчмарка — сэмплируем отдельные кадры (frame extraction). Единственный публичный датасет с такой детализацией классов насилия.
- **Загрузка:**
  ```python
  ds = load_dataset("jherng/xd-violence")
  # Для бенчмарка извлекать кадры через: cv2.VideoCapture + uniform sampling
  ```

### 2.3 `Multimodal-Fatima/violence-detection-dataset`
- **URL:** https://huggingface.co/datasets/Multimodal-Fatima/violence-detection-dataset
- **Доступ:** публичный, без токена
- **Размер:** ~2 000 изображений
- **Описание:** Бинарный (violent/non-violent), реальные сцены. Создан специально для image-level классификации (не видео). Хорошее соотношение классов 50/50.
- **Загрузка:**
  ```python
  ds = load_dataset("Multimodal-Fatima/violence-detection-dataset")
  ```

### 2.4 `Kaggle: Real Life Violence Situations` ⚠️
- **URL:** https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
- **Доступ:** требует Kaggle API token (`KAGGLE_USERNAME` + `KAGGLE_KEY`)
- **Размер:** ~2 000 видео-клипов (Violence / NonViolence)
- **Описание:** Классический датасет — уличные драки, нападения, реальные ситуации. Золотой стандарт для violence detection. Для бенчмарка — извлекать ключевые кадры.
- **Загрузка:**
  ```bash
  kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset
  ```

### 2.5 `Kaggle: Graphical Violence and Safe Images Dataset` ⚠️
- **URL:** https://www.kaggle.com/datasets/kartikeybartwal/graphical-violence-and-safe-images-dataset
- **Доступ:** требует Kaggle API token
- **Размер:** средний (~5k изображений)
- **Описание:** Уже нарезанные изображения (не видео), что удобнее для image-level бенчмарка. Включает gore, war imagery, brutal content рядом с safe-негативами.
- **Загрузка:**
  ```bash
  kaggle datasets download -d kartikeybartwal/graphical-violence-and-safe-images-dataset
  ```

---

## 3. Класс: HATEFUL / EXTREMIST

### 3.1 `neuralcatcher/hateful_memes` (Facebook AI Hateful Memes Challenge)
- **URL:** https://huggingface.co/datasets/neuralcatcher/hateful_memes
- **Доступ:** публичный, без токена
- **Размер:** ~10 000 мемов (train/dev/test splits с seen/unseen)
- **Описание:** Канонический бенчмарк Facebook AI (NeurIPS 2020). Мемы с текстом поверх изображения, бинарная разметка (hateful/not-hateful). Трудные кейсы: мем безвреден без текста, но токсичен с ним (мультимодальность). Метрика: AUROC.
- **Загрузка:**
  ```python
  ds = load_dataset("neuralcatcher/hateful_memes")
  ```

### 3.2 `limjiayi/hateful_memes_expanded`
- **URL:** https://huggingface.co/datasets/limjiayi/hateful_memes_expanded
- **Доступ:** публичный, без токена
- **Размер:** расширенная версия (~12 000 мемов с augmented splits)
- **Описание:** Расширение оригинального датасета FB с `dev_unseen` и `test_unseen` сплитами — именно они тестируют генерализацию на "новые" типы hate speech, не виденные при обучении. Критически важен для adversarial robustness.
- **Загрузка:**
  ```python
  ds = load_dataset("limjiayi/hateful_memes_expanded")
  ```

### 3.3 `QCRI/MemeLens` (подмножество: `toxic_ru__Toxic_Memes_Detection_Dataset`) 🇷🇺
- **URL:** https://huggingface.co/datasets/QCRI/MemeLens
- **Доступ:** публичный, без токена (MIT license)
- **Размер:** 6 460 русскоязычных мемов (subset `toxic_ru`); всего в MemeLens 46 subset'ов по 15+ языкам
- **Описание:** **Единственный публично доступный датасет токсичных русских мемов.** Изображения с русским текстом, размечены как `toxic` / `not-toxic`. Охватывает политические, националистические и сексистские мемы из Рунета. Является частью MemeLens — крупнейшего мультиязычного датасета мемов (QCRI/Qatar Computing Research Institute).
- **Загрузка:**
  ```python
  ds = load_dataset("QCRI/MemeLens", "toxic_ru__Toxic_Memes_Detection_Dataset")
  # Для всех русских подмножеств:
  from datasets import get_dataset_config_names
  configs = get_dataset_config_names("QCRI/MemeLens")
  ru_configs = [c for c in configs if "_ru_" in c]
  ```

### 3.4 `QCRI/MemeLens` (подмножество: `Hateful_en_FHM` и др. EN)
- **URL:** https://huggingface.co/datasets/QCRI/MemeLens
- **Доступ:** публичный, без токена
- **Размер:** 11 000 мемов (FHM subset); 59 300 (MMHS subset)
- **Описание:** Внутри MemeLens несколько EN hate speech датасетов: `Hateful_en_FHM` (Facebook Hateful Memes), `Hateful_en__MMHS` (Multimodal Hate Speech), `Harmful_en__HarMeme`, `Misogyny_en__MAMI`. Удобно загружать все через единый интерфейс MemeLens.
- **Загрузка:**
  ```python
  ds_fhm   = load_dataset("QCRI/MemeLens", "Hateful_en_FHM")
  ds_mmhs  = load_dataset("QCRI/MemeLens", "Hateful_en__MMHS")
  ds_harm  = load_dataset("QCRI/MemeLens", "Harmful_en__HarMeme")
  ds_mami  = load_dataset("QCRI/MemeLens", "misogynous_en__MAMI")
  ```

### 3.5 `NiGuLa/Russian_Inappropriate_Messages` 🇷🇺
- **URL:** https://huggingface.co/datasets/NiGuLa/Russian_Inappropriate_Messages
- **Доступ:** публичный, без токена
- **Размер:** ~10 000 текстовых постов (с топиками: политика, секс, оружие и др.)
- **Описание:** Российские сообщения с разметкой на "inappropriate topics" (не просто мат, но репутационно-вредный контент): politics, sex, drugs, weapons, religion, discrimination. Используется как **текстовый контекст** при мультимодальном анализе изображений с подписями/наложенным текстом.
- **Загрузка:**
  ```python
  ds = load_dataset("NiGuLa/Russian_Inappropriate_Messages")
  ```

### 3.6 `RuEthnoHate` (GitHub, Skoltech/HSE) 🇷🇺 ⚠️
- **URL:** https://github.com/hse-scila/ethnohate-project
- **Доступ:** прямая загрузка с GitHub (CSV)
- **Размер:** 12 300 текстов социальных сетей с этнической ненавистью
- **Описание:** Первый датасет этнической ненависти на русском. Охватывает ксенофобию к мигрантам, националистические высказывания. Используется совместно с изображениями при тестировании мультимодальных детекторов.
- **Загрузка:**
  ```bash
  git clone https://github.com/hse-scila/ethnohate-project
  # CSV файлы в папке data/
  ```

### 3.7 `TrustAIRLab/Hateful_Memes_in_VLM`
- **URL:** https://huggingface.co/datasets/TrustAIRLab/Hateful_Memes_in_VLM
- **Доступ:** публичный, без токена
- **Размер:** 39 мемов × 12 000+ ответов от 7 VLM
- **Описание:** Нетривиальный датасет: тестирует, как VLM (InstructBLIP, ShareGPT4V, LLaVA, CogVLM) **воспроизводят** hate content в ответ на мемы. Используется для проверки, не является ли цензор сам источником генерации хейта.
- **Загрузка:**
  ```python
  ds = load_dataset("TrustAIRLab/Hateful_Memes_in_VLM")
  ```

---

## 4. Класс: DISTURBING / SELF-HARM

### 4.1 `Qu et al. / UnsafeBench` (LAION-5B subset) ⚠️
- **URL:** https://arxiv.org/abs/2405.03486 (запрос к авторам)
- **Доступ:** только по запросу для исследований (contact: авторы из CISPA)
- **Размер:** 10 000 изображений, 11 категорий (sexual, violent, hateful, self-harm и др.)
- **Описание:** Аннотирован тремя экспертами, содержит как реальные (из LAION-5B), так и AI-generated (Lexica) изображения. Несмотря на ограничения разметки — единственный комплексный multi-class visual safety benchmark с официальной публикацией (ACM CCS 2025). Запрашивать через форму на странице arXiv.
- **Загрузка:**
  ```
  # Написать авторам: unsafebench@cispa.de (по данным публикации)
  # Либо запросить через форму на странице arXiv
  ```

### 4.2 `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
- **URL:** https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
- **Доступ:** публичный, без токена
- **Размер:** крупный (~26k примеров), текст + категории
- **Описание:** NVIDIA датасет для LLM safety guardrails. Охватывает violence, self-harm, sexual, harassment, hate speech. Хотя это текстовый датасет — категории и описания используются для расширения визуального pipeline через multimodal описания изображений.
- **Загрузка:**
  ```python
  ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
  ```

---

## 5. Класс: ADVERSARIAL

Этот класс критически важен — он тестирует устойчивость цензора к попыткам обхода.

### 5.1 Hateful Illusions Dataset (CISPA, ICCV 2025) ⚠️
- **URL:** https://arxiv.org/abs/2507.22617 (ICCV 2025 paper)
- **Доступ:** по запросу к авторам (CISPA Helmholtz Center)
- **Размер:** 1 571 AI-generated оптических иллюзий с embedded hate messages (из 1 860 попыток)
- **Описание:** **Самый нетривиальный adversarial датасет.** Генерируется через Stable Diffusion + ControlNet: изображение выглядит безобидно (люди на улице), но при зуммировании/отдалении проявляется слово/символ ненависти. **Результат:** 6 классификаторов не справились (accuracy < 24.5%), 9 VLM — accuracy < 10.2%. Именно такие кейсы должны быть в adversarial части бенчмарка.
- **Загрузка:**
  ```
  # Написать авторам: yiting.qu@cispa.de
  # Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/...
  ```

### 5.2 Adversarial patches / perturbations (синтетическая генерация)
- **URL:** генерируется скриптом (нет готового датасета)
- **Доступ:** генерируется локально
- **Описание:** Для adversarial теста можно сгенерировать самостоятельно:
  1. **Pixel perturbation** — FGSM/PGD атаки на unsafe изображения (они должны остаться unsafe, но классификатор их пропускает)
  2. **Steganography** — LSB embedding текстовых hate messages в нейтральные изображения
  3. **Watermark obfuscation** — добавление шума поверх explicit content
  4. **Color space jitter** — изменение цветовых каналов для bypass детектора
- **Загрузка:**
  ```python
  # Пример FGSM для тестирования:
  from torchattacks import FGSM
  atk = FGSM(model, eps=8/255)
  adversarial_img = atk(img_tensor, label)
  ```

### 5.3 `sahajps/Meme-Sanity` (counterfactual мемы)
- **URL:** https://huggingface.co/datasets/sahajps/Meme-Sanity
- **Доступ:** публичный (auto-approve gated), без токена
- **Размер:** 2 479 нейтрализованных мемов
- **Описание:** **Adversarial safe-negative.** Каждый мем из Hateful Memes Challenge был нейтрализован через LLM — изменён текст или изображение так, что мем перестал быть hateful. Используется для проверки **false positive**: цензор не должен блокировать эти мемы, но они визуально похожи на оригинальные hateful мемы.
- **Загрузка:**
  ```python
  ds = load_dataset("sahajps/Meme-Sanity")
  # Все примеры label=0 (not-hateful), использовать как hard-negative
  ```

### 5.4 Obfuscated/censored NSFW (синтетически закрашенные)
- **Описание:** Взять unsafe изображения и применить к ним операции, которые пользователи реально используют для обхода фильтров:
  1. Цензурные полоски / mosaic blur поверх explicit частей
  2. Emoji overlay (🍆, 🍑 поверх explicit content)
  3. Grayscale conversion
  4. Downscale до 64×64 и обратно
  5. Добавление watermark текста
- **Загрузка:**
  ```python
  # Пример генерации:
  from PIL import Image, ImageFilter
  img_blurred = img.filter(ImageFilter.GaussianBlur(radius=15))
  # Применить к выборке из любого unsafe датасета
  ```

---

## 6. SAFE-негативы

### 6.1 `zh-plus/tiny-imagenet`
- **URL:** https://huggingface.co/datasets/zh-plus/tiny-imagenet
- **Доступ:** публичный, без токена
- **Размер:** 100 000 изображений, 200 классов
- **Описание:** Разнообразные бытовые сцены, природа, объекты. Основной пул safe-негативов. Важен для FPR-тестирования.
- **Загрузка:**
  ```python
  ds = load_dataset("zh-plus/tiny-imagenet")
  ```

### 6.2 `detection-datasets/coco-2017-val`
- **URL:** https://huggingface.co/datasets/detection-datasets/coco-2017-val
- **Доступ:** публичный, без токена
- **Размер:** 5 000 изображений (validation split)
- **Описание:** Реальные фотографии с аннотациями объектов. Включает людей в разных контекстах — хороший тест на FP по человеческим фигурам.
- **Загрузка:**
  ```python
  ds = load_dataset("detection-datasets/coco-2017-val")
  ```

### 6.3 `wikimedia/wit_base` (Wikipedia Image-Text)
- **URL:** https://huggingface.co/datasets/wikimedia/wit_base
- **Доступ:** публичный, без токена
- **Размер:** 37 800 000 примеров (брать subset)
- **Описание:** Изображения из Wikipedia статей — содержит медицинские иллюстрации, исторические фото, произведения искусства (включая классическое ню). Критически важен для тестирования edge cases: цензор не должен блокировать Венеру Милосскую или медицинскую анатомию.
- **Загрузка:**
  ```python
  ds = load_dataset("wikimedia/wit_base", split="train[:5000]")
  ```

### 6.4 WikiArt (исторические произведения искусства) ⚠️
- **URL:** https://huggingface.co/datasets/Artificio/WikiArt
- **Доступ:** публичный, без токена
- **Размер:** ~80 000 картин, 27 стилей
- **Описание:** Классическая живопись, включая ню (Рубенс, Роден, Модильяни). Это важнейший edge case: исторические картины не должны блокироваться. Если цензор блокирует WikiArt — у него высокий FPR.
- **Загрузка:**
  ```python
  ds = load_dataset("Artificio/WikiArt")
  ```

### 6.5 `marmal88/skin_lesion` (медицинские изображения)
- **URL:** https://huggingface.co/datasets/marmal88/skin_lesion
- **Доступ:** публичный, без токена
- **Размер:** ~10 000 дерматологических изображений
- **Описание:** Медицинские фото кожи. Тест на FP для медконтента: цензор не должен блокировать дерматологию, несмотря на то что это изображения тела.
- **Загрузка:**
  ```python
  ds = load_dataset("marmal88/skin_lesion")
  ```

### 6.6 `keremberke/chest-xray-classification`
- **URL:** https://huggingface.co/datasets/keremberke/chest-xray-classification
- **Доступ:** публичный, без токена
- **Размер:** ~5 800 рентген-снимков
- **Описание:** Рентген грудной клетки. Тест: цензор не должен блокировать медицинские снимки тела.
- **Загрузка:**
  ```python
  ds = load_dataset("keremberke/chest-xray-classification")
  ```

---

## 7. Рекомендованный состав тестовой выборки

```
benchmark_test/
├── unsafe/                         (~3 500 изображений)
│   ├── sexual/          500        deepghs/nsfw_detect (porn+hentai) + yesidobyte/nsfw1024
│   ├── sexual_ai_gen/   300        wallstoneai/civitai + deepghs/nsfw_detect (drawing)
│   ├── violence/        400        Multimodal-Fatima/violence-detection + Kaggle RLVS кадры
│   ├── gore/            200        NeuralShell/Gore-Blood-Dataset
│   ├── hateful_en/      500        neuralcatcher/hateful_memes (test_seen + test_unseen)
│   ├── hateful_ru/      500        QCRI/MemeLens (toxic_ru subset)
│   └── disturbing/      300        UnsafeBench (disturbing subset, по запросу)
│
├── adversarial/                    (~600 изображений)
│   ├── optical_illusions/ 200      Hateful Illusions Dataset (по запросу)
│   ├── counterfactual_memes/ 200   sahajps/Meme-Sanity (hard negative — label=0)
│   └── obfuscated/       200       Синтетически обфусцированный unsafe контент
│
└── safe_negative/                  (~3 500 изображений)
    ├── everyday/        1000       zh-plus/tiny-imagenet (random sample)
    ├── coco_people/      500       detection-datasets/coco-2017-val (person class)
    ├── art_classical/    500       Artificio/WikiArt (ню/телесные образы из классики)
    ├── medical/          500       marmal88/skin_lesion + keremberke/chest-xray
    ├── benign_memes/     500       sahajps/Meme-Sanity (нейтрализованные мемы)
    └── wikipedia/        500       wikimedia/wit_base (random subset)
```

**Итоговое соотношение: ~3 500 unsafe : 3 500 safe : 600 adversarial ≈ 1:1 (без adversarial)**  
При подсчёте метрик adversarial обрабатывается отдельно.

---

## 8. Шаблон загрузки

```python
"""
censor_benchmark_loader.py
Загрузка всех датасетов для бенчмарка цензор-модуля.
Требования: pip install datasets kaggle Pillow tqdm
Переменные окружения: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY
"""

import os
from datasets import load_dataset
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")

# ── SEXUAL ───────────────────────────────────────────────────
def load_sexual():
    return {
        "nsfw1024": load_dataset("yesidobyte/nsfw1024"),
        "civitai":  load_dataset("wallstoneai/civitai-top-nsfw-images-with-metadata"),
        "deepghs":  load_dataset("deepghs/nsfw_detect", token=HF_TOKEN),  # 🔒
        "amaye15":  load_dataset("amaye15/NSFW"),
    }

# ── VIOLENCE ─────────────────────────────────────────────────
def load_violence():
    return {
        "xd_violence":  load_dataset("jherng/xd-violence"),
        "vio_fatima":   load_dataset("Multimodal-Fatima/violence-detection-dataset"),
        "gore_blood":   load_dataset("NeuralShell/Gore-Blood-Dataset-v1.0", token=HF_TOKEN),  # 🔒
    }

# ── HATEFUL ───────────────────────────────────────────────────
def load_hateful():
    return {
        # EN мемы
        "fb_hateful":       load_dataset("neuralcatcher/hateful_memes"),
        "fb_hateful_exp":   load_dataset("limjiayi/hateful_memes_expanded"),
        "memelens_fhm":     load_dataset("QCRI/MemeLens", "Hateful_en_FHM"),
        "memelens_mmhs":    load_dataset("QCRI/MemeLens", "Hateful_en__MMHS"),
        "memelens_harm":    load_dataset("QCRI/MemeLens", "Harmful_en__HarMeme"),
        "memelens_mami":    load_dataset("QCRI/MemeLens", "misogynous_en__MAMI"),
        # RU мемы и текст
        "memelens_ru_toxic": load_dataset("QCRI/MemeLens", "toxic_ru__Toxic_Memes_Detection_Dataset"),
        "ru_inappropriate":  load_dataset("NiGuLa/Russian_Inappropriate_Messages"),
    }

# ── ADVERSARIAL ───────────────────────────────────────────────
def load_adversarial():
    return {
        # Hard-negatives (визуально похожи на hate, но label=0)
        "meme_sanity": load_dataset("sahajps/Meme-Sanity"),
        # Hateful Illusions — запрашивать отдельно, путь к локальной папке:
        # "hateful_illusions": load_from_local("data/hateful_illusions/"),
    }

# ── SAFE NEGATIVES ────────────────────────────────────────────
def load_safe():
    return {
        "tiny_imagenet":  load_dataset("zh-plus/tiny-imagenet"),
        "coco_val":       load_dataset("detection-datasets/coco-2017-val"),
        "wikiart":        load_dataset("Artificio/WikiArt"),
        "skin_lesion":    load_dataset("marmal88/skin_lesion"),
        "chest_xray":     load_dataset("keremberke/chest-xray-classification"),
        "wikipedia_wit":  load_dataset("wikimedia/wit_base", split="train[:5000]"),
    }


# ── ПОЛНАЯ ЗАГРУЗКА ───────────────────────────────────────────
def load_all():
    print("Loading sexual datasets...")
    sexual = load_sexual()
    print("Loading violence datasets...")
    violence = load_violence()
    print("Loading hateful datasets...")
    hateful = load_hateful()
    print("Loading adversarial datasets...")
    adversarial = load_adversarial()
    print("Loading safe negative datasets...")
    safe = load_safe()
    return {
        "sexual":       sexual,
        "violence":     violence,
        "hateful":      hateful,
        "adversarial":  adversarial,
        "safe":         safe,
    }
```

---

## Статус проверки датасетов

| Датасет | Верифицирован | Доступ | Примечание |
|---|---|---|---|
| `yesidobyte/nsfw1024` | ✅ | Публичный | |
| `wallstoneai/civitai-top-nsfw-images-with-metadata` | ✅ | Публичный | |
| `deepghs/nsfw_detect` | ✅ | 🔒 HF_TOKEN | 1.8 GB zip |
| `QuixiAI/SexDrugsAndRockAndRoll` | ✅ | Публичный | Мультикатегория |
| `amaye15/NSFW` | ✅ | Публичный | |
| `NeuralShell/Gore-Blood-Dataset-v1.0` | ✅ | 🔒 HF_TOKEN | Маленький (<1k) |
| `jherng/xd-violence` | ✅ | Публичный | Видео → кадры |
| `Multimodal-Fatima/violence-detection-dataset` | ✅ | Публичный | |
| `Kaggle: Real Life Violence` | ✅ | ⚠️ Kaggle API | |
| `Kaggle: Graphical Violence` | ✅ | ⚠️ Kaggle API | |
| `neuralcatcher/hateful_memes` | ✅ | Публичный | |
| `limjiayi/hateful_memes_expanded` | ✅ | Публичный | |
| `QCRI/MemeLens` (все subsets) | ✅ | Публичный | MIT license |
| `NiGuLa/Russian_Inappropriate_Messages` | ✅ | Публичный | Текст |
| `RuEthnoHate` (GitHub) | ✅ | ⚠️ GitHub | Текст + CSV |
| `TrustAIRLab/Hateful_Memes_in_VLM` | ✅ | Публичный | VLM responses |
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | ✅ | Публичный | Текст |
| `UnsafeBench` | ✅ | ⚠️ По запросу | 11 категорий |
| `Hateful Illusions Dataset` | ✅ | ⚠️ По запросу | ICCV 2025 |
| `sahajps/Meme-Sanity` | ✅ | Публичный | Hard-negative |
| `zh-plus/tiny-imagenet` | ✅ | Публичный | |
| `detection-datasets/coco-2017-val` | ✅ | Публичный | |
| `wikimedia/wit_base` | ✅ | Публичный | Брать subset |
| `Artificio/WikiArt` | ✅ | Публичный | Edge case ню |
| `marmal88/skin_lesion` | ✅ | Публичный | Медицина |
| `keremberke/chest-xray-classification` | ✅ | Публичный | Медицина |

---

*Документ создан для передачи в Claude Code. Последнее обновление: June 2026.*
