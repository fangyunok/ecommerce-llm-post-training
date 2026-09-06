from __future__ import annotations

import re

from .schemas import RuleConstraint, RuleProduct, RuleRecommendationRequest, RuleSort


PRODUCT_PATTERN = re.compile(
    r"固态硬盘[甲乙丙]|U盘[A-C]|[ABC](?:款|电脑|平板|显示器|手机|手表)|"
    r"[MN]键盘|[PQR]款|[XYZ]充电宝|[甲乙丙](?:相机|主机|耳机)?"
)

INSTRUCTION_MARKERS = (
    "用户只要求",
    "用户要求",
    "用户",
    "没有其他偏好",
    "至少需要",
    "在满足容量",
    "预算",
    "内存至少",
    "屏幕至少",
    "至少10000mAh",
    "至少2TB",
)


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _description_part(text: str) -> str:
    positions = [text.find(marker) for marker in INSTRUCTION_MARKERS if text.find(marker) > 0]
    return text[: min(positions)] if positions else text


def _extract_product(name: str, chunk: str) -> RuleProduct:
    values: dict[str, float] = {}
    display: dict[str, str] = {}

    specs = (
        ("price", r"(\d+(?:\.\d+)?)\s*元", "元"),
        ("battery_capacity", r"(\d+(?:\.\d+)?)\s*mAh", "mAh"),
        ("weight", r"(\d+(?:\.\d+)?)\s*kg", "kg"),
        ("screen", r"(\d+(?:\.\d+)?)\s*英寸", "英寸"),
        ("refresh_rate", r"(\d+(?:\.\d+)?)\s*Hz", "Hz"),
        ("camera", r"(\d+(?:\.\d+)?)\s*MP", "MP"),
        ("storage", r"(\d+(?:\.\d+)?)\s*TB", "TB"),
        ("battery", r"续航\s*(\d+(?:\.\d+)?)\s*小时", "小时"),
    )
    for field, pattern, unit in specs:
        value = _number(pattern, chunk)
        if value is not None:
            values[field] = value
            display[field] = f"{field_label(field)}{value:g}{unit}"

    gb = _number(r"(\d+(?:\.\d+)?)\s*GB", chunk)
    if gb is not None:
        field = "memory" if "主机" in name or "内存" in chunk else "capacity"
        values[field] = gb
        display[field] = f"{field_label(field)}{gb:g}GB"

    return RuleProduct(name=name, values=values, display=display)


def field_label(field: str) -> str:
    return {
        "price": "售价",
        "battery_capacity": "容量",
        "capacity": "容量",
        "memory": "内存",
        "weight": "重量",
        "screen": "屏幕",
        "refresh_rate": "刷新率",
        "camera": "主摄",
        "storage": "容量",
        "battery": "续航",
    }[field]


def _extract_products(text: str) -> list[RuleProduct]:
    description = _description_part(text)
    matches = list(PRODUCT_PATTERN.finditer(description))
    products: list[RuleProduct] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        name = match.group(0)
        if name in seen:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        product = _extract_product(name, description[match.end() : end])
        if product.values:
            products.append(product)
            seen.add(name)
    if len(products) < 2:
        raise ValueError("无法稳定识别至少两个候选商品，请改用结构化规则接口")
    return products


def _unit_field(unit: str, text: str) -> str:
    return {
        "mAh": "battery_capacity",
        "GB": "memory" if "内存" in text else "capacity",
        "TB": "storage",
        "英寸": "screen",
        "Hz": "refresh_rate",
        "MP": "camera",
        "kg": "weight",
    }[unit]


def _extract_constraints(text: str) -> list[RuleConstraint]:
    constraints: list[RuleConstraint] = []
    budget = _number(
        r"(?:预算(?:上限严格为|最多|不超过|只有|为)?|最多花|价格上限(?:为)?)"
        r"\s*(\d+(?:\.\d+)?)\s*元",
        text,
    )
    if budget is not None:
        constraints.append(RuleConstraint(field="price", operator="le", value=budget))

    threshold_pattern = re.compile(
        r"(?:至少(?:需要)?|不得低于|不能低于|不低于)\s*"
        r"(\d+(?:\.\d+)?)\s*(mAh|GB|TB|英寸|Hz|MP)",
        flags=re.IGNORECASE,
    )
    for match in threshold_pattern.finditer(text):
        constraints.append(
            RuleConstraint(
                field=_unit_field(match.group(2), text),
                operator="ge",
                value=float(match.group(1)),
            )
        )

    above_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(mAh|GB|TB|英寸|Hz|MP)\s*(?:及以上|以上)",
        flags=re.IGNORECASE,
    )
    for match in above_pattern.finditer(text):
        constraint = RuleConstraint(
            field=_unit_field(match.group(2), text),
            operator="ge",
            value=float(match.group(1)),
        )
        if constraint not in constraints:
            constraints.append(constraint)

    weight = _number(r"重量(?:不得超过|不超过|不高于|至多)\s*(\d+(?:\.\d+)?)\s*kg", text)
    if weight is not None:
        constraints.append(RuleConstraint(field="weight", operator="le", value=weight))
    return constraints


def _extract_sort(text: str) -> RuleSort:
    if re.search(r"续航(?:最长|更长|最久)|优先续航", text):
        return RuleSort(field="battery", direction="desc")
    if re.search(r"像素最高|主摄最高", text):
        return RuleSort(field="camera", direction="desc")
    if re.search(r"价格(?:最低|较低|低)|最低价|最便宜|更便宜|低价|从低到高", text):
        return RuleSort(field="price", direction="asc")
    return RuleSort(field="price", direction="asc")


def extract_rule_request(text: str) -> RuleRecommendationRequest:
    products = _extract_products(text)
    constraints = _extract_constraints(text)
    sort = _extract_sort(text)
    reason_fields = [sort.field]
    reason_fields.extend(constraint.field for constraint in constraints if constraint.field != sort.field)
    return RuleRecommendationRequest(
        products=products,
        constraints=constraints,
        sort=sort,
        reason_fields=list(dict.fromkeys(reason_fields)),
    )
