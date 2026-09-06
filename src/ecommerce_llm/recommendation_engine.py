from __future__ import annotations

import math

from .schemas import RuleConstraint, RuleProduct, RuleRecommendationRequest, RuleRecommendationResponse


OPERATORS = {
    "le": lambda actual, expected: actual <= expected,
    "lt": lambda actual, expected: actual < expected,
    "ge": lambda actual, expected: actual >= expected,
    "gt": lambda actual, expected: actual > expected,
    "eq": lambda actual, expected: math.isclose(actual, expected),
}


def recommend_by_rules(request: RuleRecommendationRequest) -> RuleRecommendationResponse:
    eligible: list[RuleProduct] = []
    missing_fields: set[str] = set()

    for product in request.products:
        satisfies_all = True
        for constraint in request.constraints:
            actual = product.values.get(constraint.field)
            if actual is None:
                missing_fields.add(constraint.field)
                satisfies_all = False
                break
            if not OPERATORS[constraint.operator](actual, constraint.value):
                satisfies_all = False
                break
        if satisfies_all:
            eligible.append(product)

    if not eligible:
        if missing_fields:
            fields = "、".join(sorted(missing_fields))
            answer = f"信息不足，缺少{fields}，无法判断或推荐。"
        else:
            answer = "目前没有合适商品，所有候选商品都不满足硬性条件。"
        return RuleRecommendationResponse(
            selected_product=None,
            eligible_products=[],
            answer=answer,
        )

    sortable = [product for product in eligible if request.sort.field in product.values]
    if not sortable:
        return RuleRecommendationResponse(
            selected_product=None,
            eligible_products=[product.name for product in eligible],
            answer=f"信息不足，缺少{request.sort.field}，无法完成排序推荐。",
        )

    reverse = request.sort.direction == "desc"
    selected = sorted(
        sortable,
        key=lambda product: (product.values[request.sort.field], product.name),
        reverse=reverse,
    )[0]
    facts = [selected.display[field] for field in request.reason_fields if field in selected.display]
    reason = "，".join(facts)
    answer = f"推荐{selected.name}。{reason}。" if reason else f"推荐{selected.name}。"
    return RuleRecommendationResponse(
        selected_product=selected.name,
        eligible_products=[product.name for product in eligible],
        answer=answer,
    )
