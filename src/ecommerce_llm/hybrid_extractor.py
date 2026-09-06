from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .constraint_extractor import extract_rule_request
from .schemas import ChatMessage, ChatRequest, ChatResponse, ExtractionTrace, RuleRecommendationRequest


class Generator(Protocol):
    def generate(self, request: ChatRequest) -> ChatResponse: ...


SYSTEM_PROMPT = """你是电商约束抽取器，只输出一个JSON对象，不解释且不使用Markdown。
字段为products、constraints、sort、tie_breakers、reason_fields。商品含name、values、display；约束含field、operator、value。
operator只能为le/lt/ge/gt/eq，direction只能为asc/desc，数值必须是JSON数字。不得编造原文没有的商品或参数。"""


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        raise ValueError("大模型抽取结果必须是JSON对象")
    raise ValueError("大模型没有返回有效JSON对象")


@dataclass
class HybridConstraintExtractor:
    generator: Generator
    enable_llm_fallback: bool = False

    def extract(self, text: str) -> tuple[RuleRecommendationRequest, ExtractionTrace]:
        try:
            return extract_rule_request(text), ExtractionTrace(source="deterministic_parser")
        except ValueError as rule_error:
            if not self.enable_llm_fallback:
                raise ValueError(f"规则抽取失败且大模型回退未启用：{rule_error}") from rule_error
            response = self.generator.generate(ChatRequest(
                messages=[ChatMessage(role="system", content=SYSTEM_PROMPT), ChatMessage(role="user", content=text)],
                max_new_tokens=768,
                do_sample=False,
            ))
            try:
                extracted = RuleRecommendationRequest.model_validate(parse_json_object(response.message.content))
            except Exception as exc:
                raise ValueError(f"大模型回退输出未通过Schema校验：{exc}") from exc
            return extracted, ExtractionTrace(source="llm_fallback", rule_error=str(rule_error))
