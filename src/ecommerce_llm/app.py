from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .config import settings
from .constraint_extractor import extract_rule_request
from .model_service import ModelService
from .recommendation_engine import recommend_by_rules
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RuleRecommendationRequest,
    RuleRecommendationResponse,
    NaturalLanguageRecommendationRequest,
    NaturalLanguageRecommendationResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
model_service = ModelService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.preload_model:
        model_service.load()
    yield


app = FastAPI(
    title="电商大模型推理服务",
    version="0.1.0",
    description="Qwen电商导购基线服务；后续加载QLoRA/DPO模型。",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status=model_service.status,
        model=settings.model_id,
        device=model_service.device_name,
        detail=model_service.load_error,
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return model_service.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc


@app.post("/v1/rule-recommendations", response_model=RuleRecommendationResponse)
def rule_recommendations(request: RuleRecommendationRequest) -> RuleRecommendationResponse:
    """使用确定性硬约束和排序生成可审计的商品决策。"""
    return recommend_by_rules(request)


@app.post("/v1/natural-language-recommendations", response_model=NaturalLanguageRecommendationResponse)
def natural_language_recommendations(
    request: NaturalLanguageRecommendationRequest,
) -> NaturalLanguageRecommendationResponse:
    try:
        extracted = extract_rule_request(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return NaturalLanguageRecommendationResponse(
        extracted=extracted,
        recommendation=recommend_by_rules(extracted),
    )
