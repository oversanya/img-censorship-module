"""Одноразовый бенчмарк LlavaGuard-0.5B на этой (CPU-only) машине.

Цель: измерить РЕАЛЬНУЮ латентность загрузки и инференса + проверить, что выход
парсится как JSON {rating, category, rationale}. НЕ часть пайплайна — это
разведочный скрипт, можно удалить после замера.

Запуск:  .venv/Scripts/python.exe tools/bench_llavaguard.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

MODEL_ID = "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE))

POLICY = """Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment.
O2: Violence, Harm, or Cruelty.
O3: Sexual Content.
O4: Nudity Content.
O5: Criminal Planning.
O6: Weapons or Substance Abuse.
O7: Self-Harm.
O8: Animal Cruelty.
O9: Disasters or Emergencies.
Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}"""

IMAGES = ["demo/woman.jpg", "demo/boobs.jpg", "demo/shluha.png"]


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 8)
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | threads={torch.get_num_threads()}")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, cache_dir=str(CACHE)
    )
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"[load] processor+model (download+init): {load_s:.1f}s")

    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": POLICY}]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    import re

    def lenient(gen: str):
        """rating/category стоят первыми в JSON — вытащим даже из обрезанного вывода."""
        r = re.search(r'"rating"\s*:\s*"([^"]+)"', gen)
        c = re.search(r'"category"\s*:\s*"([^"]+)"', gen)
        return (r.group(1) if r else None), (c.group(1) if c else None)

    for rel in IMAGES:
        p = ROOT / rel
        if not p.exists():
            print(f"[skip] {rel} (not found)")
            continue
        image = Image.open(p).convert("RGB")
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        n_in = inputs["input_ids"].shape[-1]

        for max_new in (25, 200):
            t = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True)
            dt = time.perf_counter() - t
            gen = processor.decode(out[0][n_in:], skip_special_tokens=True)
            n_out = out.shape[-1] - n_in
            rating, category = lenient(gen)
            print(f"\n[{rel}] max_new={max_new} in_tokens={n_in} out_tokens={n_out} time={dt:.1f}s ({n_out/dt:.1f} tok/s)")
            print(f"  rating={rating} | category={category}")


if __name__ == "__main__":
    main()
