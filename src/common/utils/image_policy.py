"""Local image policy: GPT inputs are resized; provider responses remain byte exact."""
from __future__ import annotations
import io
from pathlib import Path

GPT_INPUT_SIZE = (1536, 1024)

def resized_upload(path: Path) -> io.BytesIO:
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB").resize(GPT_INPUT_SIZE, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = path.stem + "_input.png"
    return buffer

def prepare_input(source: Path, destination: Path, *, gpt: bool) -> list[int]:
    from PIL import Image
    if gpt:
        with resized_upload(source) as upload:
            destination.write_bytes(upload.read())
        return [0, 0, *GPT_INPUT_SIZE]
    destination.write_bytes(source.read_bytes())
    with Image.open(source) as image:
        return [0, 0, *image.size]
