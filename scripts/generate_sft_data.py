from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "你是一个严谨的电商导购助手。只依据给定商品信息回答，逐项核对用户的硬性要求；"
    "信息不足时明确说明，不编造商品参数。"
)
PREFIXES = ["星云", "远山", "清风", "青禾", "极光", "云帆", "松影", "晨曦"]
FEATURES = ["主动降噪", "防水", "Wi-Fi 6", "快充", "高度可调", "蓝牙5.3"]


def wrap(user: str, assistant: str, category: str) -> dict:
    return {
        "category": category,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "completion": [{"role": "assistant", "content": assistant}],
    }


def product_names(rng: random.Random) -> tuple[str, str]:
    first, second = rng.sample(PREFIXES, 2)
    suffix = rng.randint(10, 99)
    return f"{first}{suffix}款", f"{second}{suffix}款"


def budget_priority(rng: random.Random) -> dict:
    a, b = product_names(rng)
    budget = rng.randrange(180, 501, 10)
    affordable_price = rng.randrange(100, budget + 1, 10)
    expensive_price = budget + rng.randrange(20, 151, 10)
    good_value = rng.randrange(35, 61, 5)
    weak_value = rng.randrange(10, good_value, 5)
    if rng.random() < 0.5:
        rows = [(a, expensive_price, weak_value), (b, affordable_price, good_value)]
        chosen = b
    else:
        rows = [(a, affordable_price, good_value), (b, expensive_price, weak_value)]
        chosen = a
    chosen_row = next(row for row in rows if row[0] == chosen)
    user = (
        f"商品信息：{rows[0][0]}售价{rows[0][1]}元、续航{rows[0][2]}小时；"
        f"{rows[1][0]}售价{rows[1][1]}元、续航{rows[1][2]}小时。"
        f"用户预算{budget}元，并优先选择续航更长的商品。请推荐并说明理由。"
    )
    answer = (
        f"推荐{chosen}。它售价{chosen_row[1]}元，符合{budget}元预算，"
        f"续航{chosen_row[2]}小时，是预算内续航更长的选择。"
    )
    return wrap(user, answer, "预算与数值排序")


def required_feature(rng: random.Random) -> dict:
    a, b = product_names(rng)
    feature = rng.choice(FEATURES)
    budget = rng.randrange(250, 601, 10)
    a_price, b_price = rng.randrange(100, budget + 1, 10), rng.randrange(100, budget + 1, 10)
    if rng.random() < 0.5:
        supported, unsupported = a, b
    else:
        supported, unsupported = b, a
    descriptions = {
        supported: f"支持{feature}",
        unsupported: f"不支持{feature}",
    }
    prices = {a: a_price, b: b_price}
    user = (
        f"商品信息：{a}售价{a_price}元、{descriptions[a]}；"
        f"{b}售价{b_price}元、{descriptions[b]}。"
        f"用户预算{budget}元，且必须支持{feature}。请推荐。"
    )
    answer = (
        f"推荐{supported}。它售价{prices[supported]}元，符合{budget}元预算，"
        f"并且支持用户明确要求的{feature}；{unsupported}不满足该硬性要求。"
    )
    return wrap(user, answer, "必选属性")


def no_match(rng: random.Random) -> dict:
    a, b = product_names(rng)
    feature = rng.choice(FEATURES)
    budget = rng.randrange(200, 501, 10)
    a_price = budget + rng.randrange(20, 121, 10)
    b_price = rng.randrange(100, budget + 1, 10)
    user = (
        f"商品信息：{a}售价{a_price}元、支持{feature}；"
        f"{b}售价{b_price}元、不支持{feature}。"
        f"用户预算{budget}元，且必须支持{feature}。请推荐。"
    )
    answer = (
        f"目前没有合适商品。{a}支持{feature}但售价{a_price}元，超出{budget}元预算；"
        f"{b}价格符合预算但不支持{feature}，两款都未同时满足全部硬性要求。"
    )
    return wrap(user, answer, "无合适商品")


def insufficient_info(rng: random.Random) -> dict:
    item = f"{rng.choice(PREFIXES)}{rng.randint(100, 999)}款笔记本"
    price = rng.randrange(3000, 7001, 100)
    max_weight = rng.choice([1.2, 1.5, 1.8])
    min_battery = rng.choice([8, 10, 12])
    user = (
        f"商品资料只说明{item}售价{price}元，没有提供重量和续航数据。"
        f"用户要求重量低于{max_weight}kg且续航超过{min_battery}小时。是否可以推荐？"
    )
    answer = (
        f"信息不足，暂时无法推荐。现有资料缺少重量和续航数据，无法判断该商品是否满足"
        f"低于{max_weight}kg和续航超过{min_battery}小时的硬性要求。"
    )
    return wrap(user, answer, "信息不足")


def fact_summary(rng: random.Random) -> dict:
    item = f"{rng.choice(PREFIXES)}{rng.randint(10, 99)}保温杯"
    capacity = rng.choice([350, 450, 500, 600, 750])
    material = rng.choice(["304不锈钢", "316不锈钢", "陶瓷内胆"])
    hours = rng.choice([6, 8, 10, 12, 16])
    price = rng.randrange(59, 200, 10)
    user = (
        f"商品信息：{item}，容量{capacity}ml，{material}，保温{hours}小时，"
        f"售价{price}元。请完整概括核心信息，不添加资料中没有的内容。"
    )
    answer = (
        f"{item}容量为{capacity}ml，采用{material}，可保温{hours}小时，售价{price}元。"
    )
    return wrap(user, answer, "事实摘要")


GENERATORS: list[Callable[[random.Random], dict]] = [
    budget_priority,
    required_feature,
    no_match,
    insufficient_info,
    fact_summary,
]


def build_split(size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = [GENERATORS[index % len(GENERATORS)](rng) for index in range(size)]
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        assert row["prompt"][-1]["role"] == "user", index
        assert row["completion"][0]["role"] == "assistant", index
        assert row["completion"][0]["content"].strip(), index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    train = build_split(args.train_size, args.seed)
    validation = build_split(args.validation_size, args.seed + 1)
    validate(train)
    validate(validation)

    train_path = PROJECT_ROOT / "data" / "processed" / "sft_train.jsonl"
    validation_path = PROJECT_ROOT / "data" / "processed" / "sft_validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    print(f"训练集：{len(train)} 条 -> {train_path}")
    print(f"验证集：{len(validation)} 条 -> {validation_path}")
    print("类别：" + "、".join(row["category"] for row in train[:5]))


if __name__ == "__main__":
    main()

