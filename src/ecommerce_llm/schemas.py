from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    do_sample: bool = False
    temperature: float = Field(default=0.3, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.8, le=2.0)

    @model_validator(mode="after")
    def require_user_message(self) -> "ChatRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages 至少需要一条 user 消息")
        return self


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    usage: Usage
    latency_seconds: float


class HealthResponse(BaseModel):
    status: Literal["loading", "ready", "error"]
    model: str
    device: str | None = None
    detail: str | None = None


class RuleProduct(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    values: dict[str, float] = Field(default_factory=dict)
    display: dict[str, str] = Field(default_factory=dict)


class RuleConstraint(BaseModel):
    field: str = Field(min_length=1, max_length=100)
    operator: Literal["le", "lt", "ge", "gt", "eq"]
    value: float


class RuleSort(BaseModel):
    field: str = Field(min_length=1, max_length=100)
    direction: Literal["asc", "desc"]


class RuleRecommendationRequest(BaseModel):
    products: list[RuleProduct] = Field(min_length=1, max_length=100)
    constraints: list[RuleConstraint] = Field(default_factory=list, max_length=20)
    sort: RuleSort
    reason_fields: list[str] = Field(default_factory=list, max_length=20)


class RuleRecommendationResponse(BaseModel):
    selected_product: str | None
    eligible_products: list[str]
    answer: str
    decision_source: Literal["deterministic_rules"] = "deterministic_rules"
