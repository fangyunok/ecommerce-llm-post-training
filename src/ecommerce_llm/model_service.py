from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings
from .schemas import ChatRequest, ChatResponse, ChatMessage, Usage


logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    device: torch.device


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded: LoadedModel | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        # Transformers generate 会修改/使用内部缓存；先串行化请求保证正确性。
        self._generation_lock = threading.Lock()

    @property
    def status(self) -> str:
        if self._load_error:
            return "error"
        if self._loaded:
            return "ready"
        return "loading"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device_name(self) -> str | None:
        return str(self._loaded.device) if self._loaded else None

    def load(self) -> None:
        if self._loaded is not None:
            return
        with self._load_lock:
            if self._loaded is not None:
                return
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                logger.info("Loading %s on %s with %s", self.settings.model_id, device, dtype)

                tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_id,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(device)

                if self.settings.adapter_path:
                    try:
                        from peft import PeftModel
                    except ImportError as exc:
                        raise RuntimeError("加载LoRA Adapter需要安装peft") from exc
                    model = PeftModel.from_pretrained(model, self.settings.adapter_path)

                model.eval()
                self._loaded = LoadedModel(tokenizer=tokenizer, model=model, device=device)
                self._load_error = None
                logger.info("Model ready")
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Model loading failed")
                raise

    def generate(self, request: ChatRequest) -> ChatResponse:
        self.load()
        assert self._loaded is not None
        loaded = self._loaded

        messages = [message.model_dump() for message in request.messages]
        if not any(message["role"] == "system" for message in messages):
            messages.insert(
                0,
                {"role": "system", "content": self.settings.default_system_prompt},
            )

        model_inputs = loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(loaded.device)

        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        if prompt_tokens > self.settings.max_input_tokens:
            raise ValueError(
                f"输入为 {prompt_tokens} tokens，超过限制 "
                f"{self.settings.max_input_tokens}"
            )

        generation_kwargs: dict = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "repetition_penalty": request.repetition_penalty,
            "use_cache": True,
            "pad_token_id": loaded.tokenizer.eos_token_id,
        }
        if request.do_sample:
            generation_kwargs.update(
                temperature=request.temperature,
                top_p=request.top_p,
            )

        started_at = time.perf_counter()
        with self._generation_lock, torch.inference_mode():
            output_ids = loaded.model.generate(**model_inputs, **generation_kwargs)
        latency = time.perf_counter() - started_at

        generated_ids = output_ids[0, prompt_tokens:]
        answer = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.shape[-1])

        return ChatResponse(
            model=self.settings.model_id,
            message=ChatMessage(role="assistant", content=answer),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_seconds=round(latency, 3),
        )


