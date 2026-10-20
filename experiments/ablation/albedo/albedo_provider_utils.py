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
from common.utils.env import load_project_env as _load_project_env
_load_project_env()


import base64
import contextlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from openai import AzureOpenAI, DefaultHttpxClient, OpenAI
except Exception as e:
    AzureOpenAI = None
    DefaultHttpxClient = None
    OpenAI = None
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


PROVIDER_CHOICES = ("doubao", "wan", "qwen2", "gpt", "gpt2", "gemini")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DOUBAO_BASE_URL = os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_ANALYSIS_MODEL = "doubao-seed-2-0-pro-260215"
DOUBAO_ALBEDO_MODEL = "doubao-seedream-5-0-260128"

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
QWEN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_ANALYSIS_MODEL = "qwen-vl-max-latest"
WAN_ALBEDO_MODEL = "wan2.7-image"
QWEN_IMAGE_ALBEDO_MODEL = "qwen-image-2.0"
# Qwen and Wan are distinct providers.
QWEN_ALBEDO_MODEL = QWEN_IMAGE_ALBEDO_MODEL
QWEN_DEFAULT_OUTPUT_WIDTH = 1280
QWEN_DEFAULT_OUTPUT_HEIGHT = 960
QWEN_DEFAULT_OUTPUT_SIZE = f"{QWEN_DEFAULT_OUTPUT_WIDTH}*{QWEN_DEFAULT_OUTPUT_HEIGHT}"

AIHUBMIX_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
GPT2_AZURE_API_VERSION = "2025-04-01-preview"
GPT_ANALYSIS_MODEL = "gpt-4.1-mini"
GPT_ALBEDO_MODEL = "gpt-image-1.5"
GPT2_ALBEDO_MODEL = "gpt-image-2"
GEMINI_ALBEDO_MODEL = "gemini-3-pro-image-preview"
GPT_ALBEDO_QUALITY = "medium"
GPT_SUPPORTED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


@dataclass
class ProviderRuntime:
    provider: str
    analysis_model: str
    albedo_model: str
    analysis_base_url: str
    albedo_base_url: str
    analysis_api_key: str
    albedo_api_key: str
    analysis_client: Any
    analysis_client_kind: str
    albedo_client: Any
    albedo_client_kind: str
    gpt_backend: str = ""
    gpt_api_version: Optional[str] = None
    albedo_quality: str = GPT_ALBEDO_QUALITY
    generation_mode: str = "edit"


def add_provider_args(parser: Any, *, include_analysis: bool) -> None:
    parser.add_argument(
        "--provider",
        "--model_provider",
        dest="provider",
        type=str,
        default=os.environ.get("ALBEDO_PROVIDER", "doubao"),
        choices=PROVIDER_CHOICES,
        help="选择调用后端: doubao / wan / qwen2 / gpt / gpt2 / gemini。",
    )
    parser.add_argument("--analysis_api_key", type=str, default=None)
    parser.add_argument("--analysis_base_url", type=str, default=None)
    parser.add_argument("--albedo_api_key", type=str, default=None)
    parser.add_argument("--albedo_base_url", type=str, default=None)
    parser.add_argument(
        "--generation_mode",
        type=str,
        default=os.environ.get("ALBEDO_GENERATION_MODE", "edit"),
        choices=["generate", "edit"],
        help="图像生成后端使用；默认 edit。",
    )
    parser.add_argument(
        "--albedo_quality",
        type=str,
        default=os.environ.get("ALBEDO_QUALITY", GPT_ALBEDO_QUALITY),
        choices=["low", "medium", "high", "auto"],
        help="provider=gpt/gpt2 时生效。",
    )
    if include_analysis:
        parser.add_argument(
            "--analysis_model",
            type=str,
            default=None,
            help="留空时按 provider 自动选择默认分析模型。",
        )
    parser.add_argument(
        "--albedo_model",
        type=str,
        default=None,
        help="留空时按 provider 自动选择默认生成模型。",
    )


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


def pil_resize_bytes(path: Path, max_side: int) -> Tuple[bytes, str]:
    if Image is None:
        data = path.read_bytes()
        return data, guess_mime(path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        long_side = max(width, height)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "image/jpeg"


def file_to_data_uri(path: Path, max_side: Optional[int] = None) -> str:
    if max_side is not None:
        data, mime = pil_resize_bytes(path, max_side=max_side)
    else:
        data = path.read_bytes()
        mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def save_url_to_file(url: str, save_path: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def save_image_response(image_item: Any, save_path: Path, timeout: int = 120) -> Optional[str]:
    url = getattr(image_item, "url", None)
    if url:
        save_url_to_file(url, save_path, timeout=timeout)
        return url
    b64_json = getattr(image_item, "b64_json", None)
    if b64_json:
        save_path.write_bytes(base64.b64decode(b64_json))
        return None
    if isinstance(image_item, dict):
        if image_item.get("url"):
            save_url_to_file(image_item["url"], save_path, timeout=timeout)
            return image_item["url"]
        if image_item.get("b64_json"):
            save_path.write_bytes(base64.b64decode(image_item["b64_json"]))
            return None
    raise RuntimeError("返回结果中既没有 url，也没有 b64_json，无法保存图像")


def extract_output_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text
    if hasattr(resp, "output"):
        texts: List[str] = []
        for item in resp.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts)
    choices = getattr(resp, "choices", None)
    if choices:
        texts = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                texts.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        texts.append(part["text"])
                    else:
                        text = getattr(part, "text", None)
                        if text:
                            texts.append(text)
        if texts:
            return "\n".join(texts)
    return str(resp)


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


def summarize_http_error(response: requests.Response, max_len: int = 4000) -> str:
    try:
        body_text = json.dumps(response.json(), ensure_ascii=False)
    except Exception:
        body_text = response.text.strip()
    if len(body_text) > max_len:
        body_text = body_text[:max_len] + "...(truncated)"
    reason = response.reason or ""
    return f"HTTP {response.status_code} {reason}: {body_text}" if body_text else f"HTTP {response.status_code} {reason}"


def normalize_provider_name(provider: str) -> str:
    normalized = (provider or "doubao").strip().lower()
    if normalized not in PROVIDER_CHOICES:
        raise ValueError(f"不支持的 provider: {provider}")
    return normalized






def ensure_openai_import() -> None:
    if OpenAI is None:
        raise ImportError('未能导入 openai，请先安装: pip install openai') from _OPENAI_IMPORT_ERROR


def ensure_azure_import() -> None:
    if AzureOpenAI is None:
        raise ImportError('未能导入 openai.AzureOpenAI，请先安装: pip install openai') from _OPENAI_IMPORT_ERROR


def ensure_ark_import() -> None:
    if Ark is None:
        raise ImportError(
            '未能导入 volcenginesdkarkruntime，请先安装: pip install "volcengine-python-sdk[ark]"'
        ) from _ARK_IMPORT_ERROR


def build_openai_client(endpoint: str, api_key: str, timeout: int) -> Any:
    ensure_openai_import()
    client_kwargs = {"base_url": endpoint, "api_key": api_key, "timeout": timeout}
    if DefaultHttpxClient is not None:
        client_kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
    return OpenAI(**client_kwargs)


def build_proxy_aware_openai_client(endpoint: str, api_key: str, timeout: int) -> Any:
    ensure_openai_import()
    return OpenAI(base_url=endpoint, api_key=api_key, timeout=timeout)


def build_gpt_client(endpoint: str, api_key: str, api_version: Optional[str], timeout: int) -> Tuple[Any, str]:
    ensure_azure_import()
    if not endpoint or not api_key:
        raise ValueError("GPT requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint,
                       api_version=api_version or DEFAULT_AZURE_API_VERSION,
                       timeout=timeout), "azure"


def get_image_size(path: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    with Image.open(path) as image:
        return image.size


def resize_image_to_png_bytes(source_path: Path, size: Tuple[int, int]) -> bytes:
    if Image is None:
        raise ImportError('未能导入 Pillow，请先安装: pip install Pillow')
    with Image.open(source_path) as image:
        resized = image.convert("RGB").resize(size, resample=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return buffer.getvalue()


def resized_upload_file(source_path: Path, size: Tuple[int, int]) -> io.BytesIO:
    buffer = io.BytesIO(resize_image_to_png_bytes(source_path, size))
    buffer.name = f"{source_path.stem}_gpt_input.png"
    return buffer


def resized_data_uri(source_path: Path, size: Tuple[int, int]) -> str:
    b64 = base64.b64encode(resize_image_to_png_bytes(source_path, size)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def normalize_gpt_image_size(size: Optional[str]) -> str:
    normalized = (size or "auto").strip().lower()
    if normalized == "adaptive":
        normalized = "auto"
    if normalized not in GPT_SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"不支持的 GPT albedo_size: {size}。可选值为: {', '.join(sorted(GPT_SUPPORTED_IMAGE_SIZES))}"
        )
    return normalized


def resolve_model_input_size(source_path, requested_size):
    return (1536, 1024)


def validate_dashscope_output_size(width: int, height: int) -> None:
    pixels = width * height
    if pixels < 768 * 768 or pixels > 2048 * 2048:
        raise ValueError(f"Qwen 输出总像素不合法: {width}x{height}")
    ratio = width / float(height)
    if ratio < 1.0 / 8.0 or ratio > 8.0:
        raise ValueError(f"Qwen 输出宽高比不合法: {width}x{height}")


def extract_dashscope_image_url(response_json: Dict[str, Any]) -> str:
    for choice in response_json.get("output", {}).get("choices", []):
        message = choice.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "image" and item.get("image"):
                return item["image"]
            if item.get("image"):
                return item["image"]
    raise RuntimeError(f"Qwen 返回中未找到图像 URL: {json.dumps(response_json, ensure_ascii=False)}")


def resolve_provider_runtime(args: Any, *, analysis_enabled: bool) -> ProviderRuntime:
    provider = normalize_provider_name(args.provider)

    if provider == "doubao":
        ensure_ark_import()
        analysis_base_url = (args.analysis_base_url or args.base_url or DOUBAO_BASE_URL).strip()
        albedo_base_url = (args.albedo_base_url or args.base_url or DOUBAO_BASE_URL).strip()
        analysis_api_key = (args.analysis_api_key or args.api_key or os.environ.get("ARK_API_KEY") or "").strip()
        albedo_api_key = (args.albedo_api_key or args.api_key or os.environ.get("ARK_API_KEY") or "").strip()
        if analysis_enabled and not analysis_api_key:
            raise RuntimeError("provider=doubao 时未检测到分析 API key，请提供 --api_key / --analysis_api_key 或设置 ARK_API_KEY。")
        if not albedo_api_key:
            raise RuntimeError("provider=doubao 时未检测到生成 API key，请提供 --api_key / --albedo_api_key 或设置 ARK_API_KEY。")
        analysis_model = getattr(args, "analysis_model", None) or DOUBAO_ANALYSIS_MODEL
        albedo_model = getattr(args, "albedo_model", None) or DOUBAO_ALBEDO_MODEL
        analysis_client = build_openai_client(analysis_base_url, analysis_api_key, args.timeout) if analysis_enabled else None
        albedo_client = Ark(base_url=albedo_base_url, api_key=albedo_api_key)
        return ProviderRuntime(
            provider=provider,
            analysis_model=analysis_model,
            albedo_model=albedo_model,
            analysis_base_url=analysis_base_url,
            albedo_base_url=albedo_base_url,
            analysis_api_key=analysis_api_key,
            albedo_api_key=albedo_api_key,
            analysis_client=analysis_client,
            analysis_client_kind="openai",
            albedo_client=albedo_client,
            albedo_client_kind="ark",
            generation_mode=getattr(args, "generation_mode", "edit"),
            albedo_quality=getattr(args, "albedo_quality", GPT_ALBEDO_QUALITY),
        )

    if provider in {"wan", "qwen2"}:
        analysis_base_url = (
            args.analysis_base_url
            or os.environ.get("QWEN_BASE_URL")
            or QWEN_COMPATIBLE_BASE_URL
        ).strip()
        albedo_base_url = (args.albedo_base_url or args.base_url or os.environ.get("DASHSCOPE_BASE_URL") or QWEN_BASE_URL).strip()
        analysis_api_key = (
            args.analysis_api_key or args.api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        ).strip()
        albedo_api_key = (
            args.albedo_api_key or args.api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        ).strip()
        if analysis_enabled and not analysis_api_key:
            raise RuntimeError(f"provider={provider} 时未检测到分析 API key，请提供 --api_key / --analysis_api_key 或设置 DASHSCOPE_API_KEY。")
        if not albedo_api_key:
            raise RuntimeError(f"provider={provider} 时未检测到生成 API key，请提供 --api_key / --albedo_api_key 或设置 DASHSCOPE_API_KEY。")
        analysis_model = getattr(args, "analysis_model", None) or QWEN_ANALYSIS_MODEL
        default_albedo_model = QWEN_IMAGE_ALBEDO_MODEL if provider in {"qwen2"} else WAN_ALBEDO_MODEL
        if getattr(args, "albedo_model", None) not in (None, "", default_albedo_model):
            raise ValueError(f"provider={provider} requires {default_albedo_model}; refusing a different DashScope model")
        albedo_model = getattr(args, "albedo_model", None) or default_albedo_model
        analysis_client = build_openai_client(analysis_base_url, analysis_api_key, args.timeout) if analysis_enabled else None
        return ProviderRuntime(
            provider=provider,
            analysis_model=analysis_model,
            albedo_model=albedo_model,
            analysis_base_url=analysis_base_url,
            albedo_base_url=albedo_base_url,
            analysis_api_key=analysis_api_key,
            albedo_api_key=albedo_api_key,
            analysis_client=analysis_client,
            analysis_client_kind="openai",
            albedo_client=None,
            albedo_client_kind="dashscope",
            generation_mode=getattr(args, "generation_mode", "edit"),
            albedo_quality=getattr(args, "albedo_quality", GPT_ALBEDO_QUALITY),
        )

    if provider == "gemini":
        analysis_base_url = (args.analysis_base_url or os.environ.get("ARK_BASE_URL") or DOUBAO_BASE_URL).strip()
        albedo_base_url = (
            args.albedo_base_url
            or args.base_url
            or os.environ.get("AIHUBMIX_BASE_URL")
            or AIHUBMIX_BASE_URL
        ).strip()
        albedo_api_key = (
            args.albedo_api_key
            or args.api_key
            or os.environ.get("AIHUBMIX_API_KEY")
            or ""
        ).strip()
        if not albedo_api_key:
            raise RuntimeError("provider=gemini 时未检测到 API key，请提供 --albedo_api_key/--api_key 或设置 AIHUBMIX_API_KEY。")
        analysis_api_key = (args.analysis_api_key or os.environ.get("ARK_API_KEY") or "").strip()
        if analysis_enabled and not analysis_api_key:
            raise RuntimeError("analysis_enabled 时未检测到 doubao 分析 API key，请提供 --analysis_api_key 或设置 ARK_API_KEY。")
        analysis_model = getattr(args, "analysis_model", None) or DOUBAO_ANALYSIS_MODEL
        albedo_model = getattr(args, "albedo_model", None) or GEMINI_ALBEDO_MODEL
        analysis_client = build_openai_client(analysis_base_url, analysis_api_key, args.timeout) if analysis_enabled else None
        albedo_client = build_proxy_aware_openai_client(albedo_base_url, albedo_api_key, args.timeout)
        return ProviderRuntime(
            provider=provider,
            analysis_model=analysis_model,
            albedo_model=albedo_model,
            analysis_base_url=analysis_base_url,
            albedo_base_url=albedo_base_url,
            analysis_api_key=analysis_api_key,
            albedo_api_key=albedo_api_key,
            analysis_client=analysis_client,
            analysis_client_kind="openai",
            albedo_client=albedo_client,
            albedo_client_kind="gemini_chat",
            generation_mode=getattr(args, "generation_mode", "edit"),
            albedo_quality=getattr(args, "albedo_quality", GPT_ALBEDO_QUALITY),
        )

    is_gpt2 = provider == "gpt2"
    gpt_backend = "azure"
    albedo_base_url = (args.albedo_base_url or args.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    shared_api_key = (args.albedo_api_key or args.api_key or os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    if not albedo_base_url or not shared_api_key:
        raise RuntimeError("GPT requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (or CLI overrides).")
    gpt_api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION
    analysis_base_url = (args.analysis_base_url or os.environ.get("ARK_BASE_URL") or DOUBAO_BASE_URL).strip()
    analysis_api_key = (args.analysis_api_key or os.environ.get("ARK_API_KEY") or "").strip()
    if analysis_enabled and not analysis_api_key:
        raise RuntimeError("Illumination analysis requires ARK_API_KEY.")
    analysis_model = getattr(args, "analysis_model", None) or DOUBAO_ANALYSIS_MODEL
    default_gpt_model = GPT2_ALBEDO_MODEL if is_gpt2 else GPT_ALBEDO_MODEL
    albedo_model = getattr(args, "albedo_model", None) or default_gpt_model
    analysis_client = build_openai_client(analysis_base_url, analysis_api_key, args.timeout) if analysis_enabled else None
    albedo_client, albedo_client_kind = build_gpt_client(albedo_base_url, shared_api_key, gpt_api_version, args.timeout)
    return ProviderRuntime(
        provider=provider,
        analysis_model=analysis_model,
        albedo_model=albedo_model,
        analysis_base_url=analysis_base_url,
        albedo_base_url=albedo_base_url,
        analysis_api_key=analysis_api_key,
        albedo_api_key=shared_api_key,
        analysis_client=analysis_client,
        analysis_client_kind="openai",
        albedo_client=albedo_client,
        albedo_client_kind=albedo_client_kind,
        gpt_backend=gpt_backend,
        gpt_api_version=gpt_api_version,
        generation_mode=getattr(args, "generation_mode", "edit"),
        albedo_quality=getattr(args, "albedo_quality", GPT_ALBEDO_QUALITY),
    )


def run_analysis_request(
    runtime: ProviderRuntime,
    image_path: Path,
    prompt: str,
    detail: str,
    analysis_max_side: int,
) -> str:
    if runtime.analysis_client is None:
        raise RuntimeError("当前 runtime 未初始化 analysis client")
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_url: Dict[str, Any] = {"url": file_to_data_uri(image_path)}
    if detail in {"low", "high"}:
        image_url["detail"] = detail
    content.append({"type": "image_url", "image_url": image_url})
    response = runtime.analysis_client.chat.completions.create(
        model=runtime.analysis_model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
    )
    return extract_output_text(response)


def run_ark_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
) -> Any:
    image_data = [file_to_data_uri(path) for path in image_paths]
    normalized_size = (size or "").strip().lower()
    if not normalized_size or normalized_size == "adaptive":
        normalized_size = "2k"
    if "seedream" in runtime.albedo_model.lower():
        return runtime.albedo_client.images.generate(
            model=runtime.albedo_model,
            prompt=prompt,
            image=image_data,
            size=normalized_size,
            watermark=watermark,
        )
    try:
        return runtime.albedo_client.images.generate(
            model=runtime.albedo_model,
            prompt=prompt,
            image=image_data if len(image_data) > 1 else image_data[0],
            seed=seed,
            guidance_scale=guidance_scale,
            size=normalized_size,
            watermark=watermark,
        )
    except TypeError:
        return runtime.albedo_client.images.generate(
            model=runtime.albedo_model,
            prompt=prompt,
            image=image_data[-1],
            size=normalized_size,
            watermark=watermark,
        )


def run_dashscope_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    watermark: bool,
    seed: int,
    timeout: int,
) -> Dict[str, Any]:
    validate_dashscope_output_size(QWEN_DEFAULT_OUTPUT_WIDTH, QWEN_DEFAULT_OUTPUT_HEIGHT)
    content: List[Dict[str, Any]] = []
    if runtime.generation_mode == "edit":
        for image_path in image_paths:
            content.append({"image": file_to_data_uri(image_path)})
    content.append({"text": prompt})
    payload = {
        "model": runtime.albedo_model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {
            "size": QWEN_DEFAULT_OUTPUT_SIZE,
            "n": 1,
            "seed": seed,
            "watermark": watermark,
        },
    }
    response = requests.post(
        runtime.albedo_base_url.rstrip("/") + "/services/aigc/multimodal-generation/generation",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {runtime.albedo_api_key}",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(summarize_http_error(response))
    data = response.json()
    if data.get("code"):
        raise RuntimeError(f"{data.get('code')}: {data.get('message', '')}")
    return data


def extract_gemini_image_b64(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("Gemini 返回中没有 choices。")
    message = getattr(choices[0], "message", None)
    parts = getattr(message, "multi_mod_content", None) or []
    for part in parts:
        if hasattr(part, "model_dump"):
            part = part.model_dump()
        inline_data = part.get("inline_data") if isinstance(part, dict) else getattr(part, "inline_data", None)
        if hasattr(inline_data, "model_dump"):
            inline_data = inline_data.model_dump()
        data = inline_data.get("data") if isinstance(inline_data, dict) else getattr(inline_data, "data", None)
        if data:
            return data
    raise RuntimeError("Gemini 返回中没有 inline image。")


def run_gemini_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
) -> Any:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": file_to_data_uri(path)}}
        for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    response = runtime.albedo_client.chat.completions.create(
        model=runtime.albedo_model,
        messages=[
            {"role": "system", "content": "aspect_ratio=4:3"},
            {"role": "user", "content": content},
        ],
        modalities=["text", "image"],
    )
    return SimpleNamespace(data=[SimpleNamespace(b64_json=extract_gemini_image_b64(response))])


def run_gpt_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    seed: int,
) -> Any:
    requested_size = normalize_gpt_image_size(size)
    common_kwargs: Dict[str, Any] = {
        "model": runtime.albedo_model,
        "prompt": prompt,
        "quality": runtime.albedo_quality,
        "output_format": "png",
        "n": 1,
    }
    mode = (runtime.generation_mode or "edit").strip().lower()
    if mode not in {"generate", "edit"}:
        raise ValueError(f"不支持的 GPT generation_mode: {runtime.generation_mode}")
    model_input_size = resolve_model_input_size(image_paths[-1], requested_size)
    if mode == "generate":
        try:
            if len(image_paths) == 1:
                return runtime.albedo_client.images.generate(
                    image=resized_upload_file(image_paths[0], model_input_size),
                    seed=seed,
                    size=requested_size,
                    **common_kwargs,
                )
        except TypeError:
            pass
        try:
            return runtime.albedo_client.images.generate(
                images=[{"image_url": resized_data_uri(path, model_input_size)} for path in image_paths],
                seed=seed,
                size=requested_size,
                **common_kwargs,
            )
        except TypeError:
            mode = "edit"

    edit_kwargs = {**common_kwargs, "size": requested_size}
    if len(image_paths) > 1:
        with contextlib.ExitStack() as stack:
            uploads = [stack.enter_context(resized_upload_file(path, model_input_size)) for path in image_paths]
            try:
                return runtime.albedo_client.images.edit(image=uploads, **edit_kwargs)
            except TypeError as exc:
                raise RuntimeError(
                    f"{runtime.provider} 后端无法读取 B1 所需的 {len(image_paths)} 张输入图，"
                    "已中止以避免丢失 exemplar 输入。"
                ) from exc
    try:
        return runtime.albedo_client.images.edit(
            image=resized_upload_file(image_paths[0], model_input_size),
            **edit_kwargs,
        )
    except TypeError:
        fallback_kwargs = {
            **edit_kwargs,
            "images": [{"image_url": resized_data_uri(image_paths[0], model_input_size)}],
        }
        try:
            return runtime.albedo_client.images.edit(**fallback_kwargs)
        except TypeError:
            fallback_kwargs.pop("output_format", None)
            fallback_kwargs.pop("n", None)
            return runtime.albedo_client.images.edit(**fallback_kwargs)


def run_albedo_generation(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
    timeout: int,
) -> Any:
    if runtime.provider == "doubao":
        return run_ark_generation(runtime, prompt, image_paths, size, watermark, seed, guidance_scale)
    if runtime.provider in {"wan", "qwen2"}:
        return run_dashscope_generation(runtime, prompt, image_paths, watermark, seed, timeout)
    if runtime.provider == "gemini":
        return run_gemini_generation(runtime, prompt, image_paths)
    return run_gpt_generation(runtime, prompt, image_paths, size, seed)


def _is_retryable_generation_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    error_text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(
        marker in error_text
        for marker in (
            "apitimeouterror",
            "apiconnectionerror",
            "ratelimiterror",
            "internalservererror",
            "connecterror",
            "readtimeout",
            "connecttimeout",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "connection aborted",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
        )
    )


def run_albedo_generation_with_retry(
    runtime: ProviderRuntime,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    watermark: bool,
    seed: int,
    guidance_scale: float,
    timeout: int,
    request_retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> Any:
    retries = max(0, request_retries)
    for attempt in range(retries + 1):
        try:
            return run_albedo_generation(
                runtime, prompt, image_paths, size, watermark, seed, guidance_scale, timeout
            )
        except Exception as exc:
            if (
                runtime.provider not in {"gpt", "gpt2"}
                or attempt >= retries
                or not _is_retryable_generation_error(exc)
            ):
                raise
            delay = min(max(0.0, retry_backoff) * (2 ** attempt), max(0.0, retry_max_backoff))
            print(
                f"      generation 临时错误，{delay:.1f}s 后进行第 {attempt + 1}/{retries} 次重试："
                f"{exc.__class__.__name__}: {exc}"
            )
            time.sleep(delay)


def save_generated_image(runtime: ProviderRuntime, response: Any, save_path: Path, timeout: int) -> Optional[str]:
    if runtime.provider in {"wan", "qwen2"}:
        image_url = extract_dashscope_image_url(response)
        save_url_to_file(image_url, save_path, timeout=timeout)
        return image_url
    return save_image_response(response.data[0], save_path, timeout=timeout)
