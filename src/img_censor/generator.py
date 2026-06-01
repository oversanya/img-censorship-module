import hashlib
from pathlib import Path
from textwrap import wrap
from typing import Optional

from PIL import Image, ImageDraw


def create_mock_generated_image(prompt: Optional[str], output_dir: str = "outputs") -> str:
    text = prompt or "empty prompt"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"mock-generated-{digest}.png"

    image = Image.new("RGB", (1024, 768), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 976, 720), outline=(80, 96, 120), width=3)
    draw.text((80, 90), "Mock image generator output", fill=(20, 35, 60))
    draw.text((80, 135), "Prompt:", fill=(20, 35, 60))
    y = 170
    for line in wrap(text, width=62)[:12]:
        draw.text((80, y), line, fill=(20, 35, 60))
        y += 34
    draw.text((80, 680), "Replace this mock generator with real text2image/img2img service.", fill=(90, 105, 130))
    image.save(path)
    return str(path)

