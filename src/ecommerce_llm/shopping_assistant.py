from __future__ import annotations

import re
from dataclasses import dataclass

from .hybrid_extractor import HybridConstraintExtractor
from .recommendation_engine import recommend_by_rules
from .schemas import ChatMessage, ChatRequest, ShoppingAssistantRequest, ShoppingAssistantResponse


DECISION_WORDS = re.compile(r"推荐|选择|哪(?:个|款|台)|最低|最高|预算|至少|不超过|优先")


def looks_like_decision(text: str) -> bool:
    return bool(re.search(r"\d", text) and DECISION_WORDS.search(text))


@dataclass
class ShoppingAssistantService:
    extractor: HybridConstraintExtractor
    generator: object

    def handle(self, request: ShoppingAssistantRequest) -> ShoppingAssistantResponse:
        if request.mode != "chat":
            try:
                extracted, trace = self.extractor.extract(request.text)
                recommendation = recommend_by_rules(extracted)
                route = "deterministic_rules" if trace.source == "deterministic_parser" else "llm_extraction_rules"
                return ShoppingAssistantResponse(
                    answer=recommendation.answer,
                    route=route,
                    extraction=extracted,
                    recommendation=recommendation,
                )
            except ValueError:
                if request.mode == "decision" or looks_like_decision(request.text):
                    raise

        model_response = self.generator.generate(ChatRequest(
            messages=[ChatMessage(role="user", content=request.text)],
            max_new_tokens=256,
            do_sample=False,
        ))
        return ShoppingAssistantResponse(
            answer=model_response.message.content,
            route="language_model",
            usage=model_response.usage,
        )
