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
operator只能为le/lt/ge/gt/eq，direction只能为asc/desc，数值必须是JSON数字。不得编造原文没有的商品或参数。
字段名约定：价格price、重量weight（kg）、续航battery（小时）、电池容量battery_capacity（mAh）、内存memory（GB）、存储storage（TB）、屏幕screen（英寸）、刷新率refresh_rate（Hz）、主摄camera（MP）。
示例输入：甲耳机200元续航30小时，乙耳机180元续航20小时；预算190元，续航优先。
示例输出：{"products":[{"name":"甲耳机","values":{"price":200,"battery":30}},{"name":"乙耳机","values":{"price":180,"battery":20}}],"constraints":[{"field":"price","operator":"le","value":190}],"sort":{"field":"battery","direction":"desc"},"tie_breakers":[],"reason_fields":["price","battery"]}"""


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
