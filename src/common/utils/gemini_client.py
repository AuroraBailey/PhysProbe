"""Compatibility adapter for Aihubmix Gemini image Chat Completions.

It presents the small ``client.images`` surface used by the existing benchmark
scripts, while sending the original image bytes and prompt through the
OpenAI-compatible Chat Completions endpoint.
"""

from __future__ import annotations

import base64
import mimetypes
import os

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None


AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"


def _data_uri_from_bytes(data: bytes, filename: str = "") -> str:
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _as_data_uri(image: Any) -> str:
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        return _data_uri_from_bytes(Path(image).read_bytes(), image)
    if isinstance(image, dict):
        image_url = image.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str):
            return _as_data_uri(image_url)
    if hasattr(image, "read"):
        return _data_uri_from_bytes(image.read(), str(getattr(image, "name", "")))
    if isinstance(image, (bytes, bytearray)):
        return _data_uri_from_bytes(bytes(image))
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _iter_images(image: Any = None, images: Any = None) -> Iterable[Any]:
    source = images if images is not None else image
    if source is None:
        return ()
    if isinstance(source, (list, tuple)):
        return source
    return (source,)


def _extract_inline_image(response: Any) -> bytes:
    message = response.choices[0].message
    parts = getattr(message, "multi_mod_content", None) or []
    for part in parts:
        if hasattr(part, "model_dump"):
            part = part.model_dump()
        inline_data = part.get("inline_data") if isinstance(part, dict) else getattr(part, "inline_data", None)
        if inline_data:
            if hasattr(inline_data, "model_dump"):
                inline_data = inline_data.model_dump()
            data = inline_data.get("data") if isinstance(inline_data, dict) else getattr(inline_data, "data", None)
            if data:
                return base64.b64decode(data)
    raise RuntimeError("No inline image was returned by Gemini Chat Completions.")


# These compatibility options are not sent by this Chat Completions adapter.
UNUSED_IMAGE_OPTIONS = frozenset({
    "size", "quality", "image_quality", "output_format", "seed", "watermark",
    "requested_size", "albedo_size", "albedo_quality", "normal_size", "guidance_scale",
})


def gemini_request_record(record: dict[str, Any], *, aspect_ratio: str = "4:3") -> dict[str, Any]:
    """Separate configured compatibility options from effective request settings."""
    record = dict(record)
    unused = dict(record.get("unused_options", {}))
    for key in UNUSED_IMAGE_OPTIONS:
        if key in record:
            unused[key] = record.pop(key)
    record["unused_options"] = unused
    record["effective_request"] = {
        "backend": "aihubmix_chat_completions",
        "aspect_ratio": aspect_ratio,
        "modalities": ["text", "image"],
        "output_size": "provider_native",
        "seed_sent": False,
    }
    return record


def provider_request_record(record: dict[str, Any], provider: str) -> dict[str, Any]:
    """Keep other providers' parameter semantics unchanged."""
    return gemini_request_record(record) if provider == "gemini" else record


class _GeminiImages:
    def __init__(self, owner: "GeminiImageClient") -> None:
        self._owner = owner

    def _request(self, *, model: str, prompt: str, image: Any = None,
                 images: Any = None, n: int = 1, **_ignored: Any) -> Any:
        if n != 1:
            raise ValueError("The paper protocol requires exactly one image per query (n=1).")
        response = self._owner._create(model=model, prompt=prompt, image=image, images=images)
        response.request_parameters = gemini_request_record(
            _ignored, aspect_ratio=self._owner._aspect_ratio)
        return response

    edit = _request
    generate = _request


class GeminiImageClient:
    """Adapter retaining legacy image-call signatures without changing inputs."""

    def __init__(self, api_key: str, *, aspect_ratio: str, timeout: float | None = None,
                 base_url: str | None = None) -> None:
        if OpenAI is None:
            raise ImportError("Install the OpenAI SDK before running Gemini generation: pip install openai") from _OPENAI_IMPORT_ERROR
        kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url or os.environ.get("AIHUBMIX_BASE_URL") or AIHUBMIX_BASE_URL}
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = OpenAI(**kwargs)
        self._aspect_ratio = aspect_ratio
        self.images = _GeminiImages(self)

    def _create(self, *, model: str, prompt: str, image: Any = None, images: Any = None) -> Any:
        content = [
            {"type": "image_url", "image_url": {"url": _as_data_uri(item)}}
            for item in _iter_images(image=image, images=images)
        ]
        content.append({"type": "text", "text": prompt})
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"aspect_ratio={self._aspect_ratio}"},
                {"role": "user", "content": content},
            ],
            modalities=["text", "image"],
        )
        image_bytes = _extract_inline_image(response)
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(image_bytes).decode("ascii"))]
        )
