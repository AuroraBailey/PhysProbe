#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Support both direct script execution and editable package installation.
import sys as _env_sys
from pathlib import Path as _EnvPath
for _env_root in _EnvPath(__file__).resolve().parents:
    if (_env_root / "src/common/utils/env.py").is_file():
        _env_sys.path.insert(0, str(_env_root / "src"))
        _env_sys.path.insert(0, str(_env_root))
        break
from common.utils.image_policy import resized_upload
from common.utils.env import load_project_env as _load_project_env
_load_project_env()

import base64
import contextlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from openai import AzureOpenAI
except Exception as e:
    AzureOpenAI = None
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from volcenginesdkarkruntime import Ark
except Exception as e:
    Ark = None
    _ARK_IMPORT_ERROR = e
else:
    _ARK_IMPORT_ERROR = None


PROVIDER_CHOICES = ("doubao", "gpt")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_NORMAL_MODEL = "doubao-seedream-5-0-260128"


DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
GPT_NORMAL_MODEL = "gpt-image-1.5"
GPT_NORMAL_SIZE = "1536x1024"
GPT_NORMAL_QUALITY = "high"
GPT_SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


@dataclass
class ProviderRuntime:
    provider: str
    normal_model: str
    normal_base_url: str
    normal_api_key: str
    normal_client: Any
    normal_client_kind: str
    gpt_backend: str = ""
    gpt_api_version: Optional[str] = None
    normal_quality: str = GPT_NORMAL_QUALITY
    generation_mode: str = "edit"


def add_provider_args(parser: Any) -> None:
    parser.add_argument(
        "--provider",
        "--model_provider",
        dest="provider",
        type=str,
        default=os.environ.get("NORMAL_PROVIDER", "doubao"),
        choices=PROVIDER_CHOICES,
        help="选择调用后端: doubao / gpt。",
    )
    parser.add_argument("--normal_api_key", type=str, default=None)
    parser.add_argument(
        "--normal_base_url",
        "--base_url",
        dest="normal_base_url",
        type=str,
        default=None,
        help="生成 normal 的 base_url；不同 provider 会自动解析默认值。",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default=os.environ.get("NORMAL_GENERATION_MODE", "edit"),
        choices=["generate", "edit"],
        help="provider=gpt 时生效；默认 edit。",
    )
    parser.add_argument(
        "--normal_quality",
        type=str,
        default=os.environ.get("NORMAL_QUALITY", GPT_NORMAL_QUALITY),
        choices=["low", "medium", "high", "auto"],
        help="provider=gpt 时生效。",
    )


def normalize_provider_name(provider: str) -> str:
    normalized = (provider or "doubao").strip().lower()
    if normalized not in PROVIDER_CHOICES:
        raise ValueError(f"不支持的 provider: {provider}")
    return normalized


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def file_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def save_url_to_file(url: str, save_path: Path, timeout: int) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int) -> Optional[str]:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout)
        return url
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return None
    if isinstance(image_item, dict):
        if image_item.get("url"):
            save_url_to_file(image_item["url"], save_path, timeout)
            return image_item["url"]
        if image_item.get("b64_json"):
            save_path.write_bytes(base64.b64decode(image_item["b64_json"]))
            return None
    raise RuntimeError("Image response contains neither url nor b64_json.")


def summarize_http_error(response: requests.Response, max_len: int = 4000) -> str:
    try:
        body_text = json.dumps(response.json(), ensure_ascii=False)
    except Exception:
        body_text = response.text.strip()
    if len(body_text) > max_len:
        body_text = body_text[:max_len] + "...(truncated)"
    reason = response.reason or ""
    return f"HTTP {response.status_code} {reason}: {body_text}" if body_text else f"HTTP {response.status_code} {reason}"


def extract_error_code(exc: Exception) -> str:
    text = str(exc)
    for pattern in (r"'code':\s*'([^']+)'", r'"code":\s*"([^"]+)"'):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return exc.__class__.__name__


def should_skip_image_error(exc: Exception) -> bool:
    text = str(exc)
    patterns = [
        "InputTextSensitiveContentDetected",
        "InputImageSensitiveContentDetected",
        "SensitiveContentDetected",
        "sensitive information",
        "sensitive content",
        "moderation_blocked",
        "rejected by the safety system",
        "image_generation_user_error",
    ]
    return any(pattern in text for pattern in patterns)


def ensure_ark_import() -> None:
    if Ark is None:
        raise ImportError(
            '未能导入 volcenginesdkarkruntime，请先安装: pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR




def ensure_azure_import() -> None:
    if AzureOpenAI is None:
        raise ImportError('未能导入 openai.AzureOpenAI，请先安装: pip install openai') from _OPENAI_IMPORT_ERROR






def build_gpt_client(endpoint: str, api_key: str, api_version: Optional[str], timeout: int) -> Tuple[Any, str]:
    if not endpoint or not api_key:
        raise ValueError("GPT requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    ensure_azure_import()
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint,
                       api_version=api_version or DEFAULT_AZURE_API_VERSION,
                       timeout=timeout), "azure"


def resolve_provider_runtime(args: Any) -> ProviderRuntime:
    provider = normalize_provider_name(args.provider)

    if provider == "doubao":
        ensure_ark_import()
        api_key = (args.normal_api_key or args.api_key or os.environ.get("ARK_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("provider=doubao 时未检测到 API key，请提供 --api_key / --normal_api_key 或设置 ARK_API_KEY。")
        base_url = (args.normal_base_url or os.environ.get("ARK_BASE_URL") or DOUBAO_BASE_URL).strip()
        normal_model = getattr(args, "normal_model", None) or DOUBAO_NORMAL_MODEL
        return ProviderRuntime(
            provider=provider,
            normal_model=normal_model,
            normal_base_url=base_url,
            normal_api_key=api_key,
            normal_client=Ark(base_url=base_url, api_key=api_key),
            normal_client_kind="ark",
            generation_mode="generate",
            normal_quality=getattr(args, "normal_quality", GPT_NORMAL_QUALITY),
        )

    base_url = (args.normal_base_url or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    api_key = (args.normal_api_key or args.api_key or os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    if not base_url or not api_key:
        raise RuntimeError("GPT requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (or CLI overrides).")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION
    normal_model = getattr(args, "normal_model", None) or GPT_NORMAL_MODEL
    normal_client, client_kind = build_gpt_client(base_url, api_key, api_version, args.timeout)
    return ProviderRuntime(
        provider=provider,
        normal_model=normal_model,
        normal_base_url=base_url,
        normal_api_key=api_key,
        normal_client=normal_client,
        normal_client_kind=client_kind,
        gpt_backend="azure",
        gpt_api_version=api_version,
        generation_mode=getattr(args, "generation_mode", "edit"),
        normal_quality=getattr(args, "normal_quality", GPT_NORMAL_QUALITY),
    )


def normalize_doubao_size(size: Optional[str]) -> str:
    normalized = (size or "adaptive").strip().lower()
    if normalized in {"", "adaptive", "source"}:
        return "2k"
    return normalized








def normalize_gpt_size(size: Optional[str]) -> str:
    if not size:
        return GPT_NORMAL_SIZE
    normalized = size.strip().lower()
    if normalized in {"adaptive", "source"}:
        return GPT_NORMAL_SIZE
    if normalized not in GPT_SUPPORTED_IMAGE_SIZES:
        raise ValueError(f"不支持的 GPT normal_size: {size}")
    return normalized




def run_doubao_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    watermark: bool,
) -> Any:
    image_uris = [file_to_data_uri(path) for path in image_paths]
    normalized_size = normalize_doubao_size(size)
    if "seedream" in runtime.normal_model.lower():
        return runtime.normal_client.images.generate(
            model=runtime.normal_model,
            prompt=prompt,
            image=image_uris,
            size=normalized_size,
            watermark=watermark,
        )
    try:
        return runtime.normal_client.images.generate(
            model=runtime.normal_model,
            prompt=prompt,
            image=image_uris if len(image_uris) > 1 else image_uris[0],
            size=normalized_size,
            watermark=watermark,
        )
    except TypeError as exc:
        raise RuntimeError("Normal generation requires preserving the complete query-first input list; backend rejected the request.") from exc




def run_gpt_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
) -> Any:
    if not image_paths:
        raise ValueError("image_paths 不能为空")
    normalized_size = normalize_gpt_size(size)

    def _edit_with_fallback(image_arg: Any) -> Any:
        base_kwargs: Dict[str, Any] = {
            "model": runtime.normal_model,
            "image": image_arg,
            "prompt": prompt,
            "n": 1,
        }
        fallback_variants: List[Dict[str, Any]] = [
            {"size": normalized_size, "quality": runtime.normal_quality, "output_format": "png", "response_format": "b64_json"},
            {"size": normalized_size, "quality": runtime.normal_quality, "output_format": "png"},
            {"size": normalized_size, "quality": runtime.normal_quality},
            {"size": normalized_size},
            {},
        ]
        last_error: Optional[Exception] = None
        for extra_kwargs in fallback_variants:
            try:
                return runtime.normal_client.images.edit(**base_kwargs, **extra_kwargs, input_fidelity="high")
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                if "Unknown parameter" in str(exc):
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("images.edit 调用失败，且未捕获到具体错误。")

    with contextlib.ExitStack() as stack:
        image_files = [stack.enter_context(resized_upload(path)) for path in image_paths]
        image_arg: Any = image_files[0] if len(image_files) == 1 else image_files
        return _edit_with_fallback(image_arg)


def run_normal_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
    timeout: int,
) -> Any:
    del guidance_scale
    if runtime.provider == "doubao":
        return run_doubao_generation(runtime, prompt, image_paths, size, watermark)
    return run_gpt_generation(runtime, prompt, image_paths, size)


def save_generated_image(runtime: ProviderRuntime, response: Any, save_path: Path, timeout: int) -> Optional[str]:
    return save_image_response(response.data[0], save_path, timeout)
