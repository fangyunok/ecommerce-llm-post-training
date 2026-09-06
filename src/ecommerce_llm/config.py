from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_ID = (
    str(LOCAL_MODEL_PATH)
    if LOCAL_MODEL_PATH.exists()
    else "Qwen/Qwen2.5-0.5B-Instruct"
)


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
    adapter_path: str | None = os.getenv("ADAPTER_PATH") or None
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    preload_model: bool = _as_bool(os.getenv("PRELOAD_MODEL", "true"))
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "2048"))
    enable_llm_extraction_fallback: bool = _as_bool(
        os.getenv("ENABLE_LLM_EXTRACTION_FALLBACK", "false")
    )
    default_system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "你是一个严谨的电商导购助手。只依据用户提供的商品信息回答；"
        "信息不足时明确说明，不编造商品参数。",
    )


settings = Settings()
