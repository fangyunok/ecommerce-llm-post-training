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
PREFIXES = [
    "星云", "远山", "清风", "青禾", "极光", "云帆", "松影", "晨曦",
    "银杉", "海盐", "月川", "栖木",
]
FEATURES = ["主动降噪", "防水", "Wi-Fi 6", "快充", "高度可调", "蓝牙5.3"]
NEGATIONS = ["不支持{feature}", "没有{feature}", "不具备{feature}"]


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
    expensive_price = budget + rng.randrange(1, 121)
    good_value = rng.randrange(35, 61, 5)
    weak_value = rng.randrange(10, good_value, 5)
    if rng.random() < 0.5:
        rows = [(a, expensive_price, weak_value), (b, affordable_price, good_value)]
        chosen = b
    else:
        rows = [(a, affordable_price, good_value), (b, expensive_price, weak_value)]
        chosen = a
    chosen_row = next(row for row in rows if row[0] == chosen)
    templates = [
        (
            f"商品信息：{rows[0][0]}售价{rows[0][1]}元、续航{rows[0][2]}小时；"
            f"{rows[1][0]}售价{rows[1][1]}元、续航{rows[1][2]}小时。"
            f"用户预算不超过{budget}元，在预算内优先选续航更长的。请推荐。"
        ),
        (
            f"候选商品：{rows[1][0]}续航{rows[1][2]}小时，价格{rows[1][1]}元；"
            f"{rows[0][0]}续航{rows[0][2]}小时，价格{rows[0][1]}元。"
            f"最多能花{budget}元，先排除超预算商品，再比较续航。应该选哪款？"
        ),
    ]
    user = rng.choice(templates)
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
    unsupported_text = rng.choice(NEGATIONS).format(feature=feature)
    descriptions = {
        supported: f"支持{feature}",
        unsupported: unsupported_text,
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
    facts = [
        f"容量{capacity}ml",
        material,
        f"保温{hours}小时",
        f"售价{price}元",
    ]
    rng.shuffle(facts)
    user = (
        f"商品信息：{item}，{'，'.join(facts)}。"
        "请完整概括名称、容量、材质、保温时间和价格，不添加未知卖点。"
    )
    answer = (
        f"{item}容量为{capacity}ml，采用{material}，可保温{hours}小时，售价{price}元。"
    )
    return wrap(user, answer, "事实摘要")


def budget_boundary(rng: random.Random) -> dict:
    """训练等于预算可选、高于预算即淘汰的边界关系。"""
    a, b = product_names(rng)
    budget = rng.randrange(150, 501, 10)
    eligible_price = budget
    over_price = budget + rng.choice([1, 5, 10])
    eligible_score = rng.randrange(60, 86, 5)
    over_score = rng.randrange(eligible_score + 5, 101, 5)
    if rng.random() < 0.5:
        rows = [(a, eligible_price, eligible_score), (b, over_price, over_score)]
        chosen, rejected = a, b
    else:
        rows = [(a, over_price, over_score), (b, eligible_price, eligible_score)]
        chosen, rejected = b, a
    user = (
        f"{rows[0][0]}价格{rows[0][1]}元、评分{rows[0][2]}分；"
        f"{rows[1][0]}价格{rows[1][1]}元、评分{rows[1][2]}分。"
        f"预算上限是{budget}元，不能超支，在可买商品中选择评分更高的。请推荐。"
    )
    answer = (
        f"推荐{chosen}。它售价{eligible_price}元，等于{budget}元预算上限，可以购买；"
        f"{rejected}售价{over_price}元，高于预算，虽然评分更高也应排除。"
    )
    return wrap(user, answer, "预算边界")


def negative_attribute(rng: random.Random) -> dict:
    """强化不可、不支持、没有等否定信息。"""
    item = f"{rng.choice(PREFIXES)}{rng.randint(10, 99)}款"
    price = rng.randrange(99, 401, 10)
    budget = price + rng.randrange(10, 101, 10)
    feature = rng.choice(FEATURES)
    negative = rng.choice(NEGATIONS).format(feature=feature)
    user = (
        f"商品资料：{item}售价{price}元，{negative}。用户预算{budget}元，"
        f"并把{feature}作为必须满足的条件。这款商品合适吗？"
    )
    answer = (
        f"不合适。虽然{item}售价{price}元，没有超过{budget}元预算，但它{negative}，"
        f"不满足必须具备{feature}的硬性要求，因此不推荐。"
    )
    return wrap(user, answer, "否定属性判断")


def three_product_sort(rng: random.Random) -> dict:
    """让价格和容量保持绑定，再在满足容量的商品中选最低价。"""
    suffix = rng.randint(10, 99)
    names = [f"{name}{suffix}款" for name in rng.sample(PREFIXES, 3)]
    min_capacity = rng.choice([10000, 15000, 20000])
    eligible_capacity = [min_capacity, min_capacity + 5000]
    cheap_price = rng.randrange(79, 150, 10)
    expensive_price = cheap_price + rng.randrange(20, 101, 10)
    ineligible_price = max(39, cheap_price - rng.randrange(10, 41, 10))
    rows = [
        (names[0], expensive_price, eligible_capacity[1]),
        (names[1], ineligible_price, min_capacity - 5000),
        (names[2], cheap_price, eligible_capacity[0]),
    ]
    rng.shuffle(rows)
    user = (
        "三款充电宝："
        + "；".join(f"{name}售价{price}元、容量{capacity}mAh" for name, price, capacity in rows)
        + f"。容量必须达到{min_capacity}mAh，在合格商品中选择价格最低的。请推荐。"
    )
    answer = (
        f"推荐{names[2]}，售价{cheap_price}元、容量{min_capacity}mAh。"
        f"它达到容量要求，并且比另一款合格商品的{expensive_price}元更便宜；"
        f"{names[1]}虽便宜但容量不足。"
    )
    return wrap(user, answer, "多商品数值筛选")


def all_affordable_sort(rng: random.Random) -> dict:
    a, b = product_names(rng)
    budget = rng.randrange(300, 601, 10)
    a_price = rng.randrange(100, budget - 20, 10)
    b_price = rng.randrange(100, budget - 20, 10)
    a_value, b_value = rng.sample(range(20, 61, 5), 2)
    chosen = a if a_value > b_value else b
    chosen_price = a_price if chosen == a else b_price
    chosen_value = max(a_value, b_value)
    user = (
        f"{a}售价{a_price}元、续航{a_value}小时；{b}售价{b_price}元、续航{b_value}小时。"
        f"两款都不超过{budget}元预算，请选择续航更长的一款。"
    )
    answer = (
        f"推荐{chosen}。两款价格都在{budget}元预算内，{chosen}售价{chosen_price}元，"
        f"续航{chosen_value}小时，是续航更长的选择。"
    )
    return wrap(user, answer, "预算内排序")


def explicit_budget_comparison(rng: random.Random) -> dict:
    """显式教授价格与预算的大小关系，性能不能覆盖预算硬约束。"""
    a, b = product_names(rng)
    budget = rng.randrange(200, 601, 10)
    over_price = budget + rng.choice([1, 5, 20, 50])
    valid_price = budget - rng.choice([0, 10, 30, 50])
    over_value = rng.randrange(80, 121, 5)
    valid_value = rng.randrange(30, 76, 5)
    if rng.random() < 0.5:
        rows = [(a, over_price, over_value), (b, valid_price, valid_value)]
        chosen, rejected = b, a
    else:
        rows = [(a, valid_price, valid_value), (b, over_price, over_value)]
        chosen, rejected = a, b
    user = (
        f"{rows[0][0]}售价{rows[0][1]}元、性能分{rows[0][2]}；"
        f"{rows[1][0]}售价{rows[1][1]}元、性能分{rows[1][2]}。"
        f"预算最多{budget}元，绝对不能超预算，然后才比较性能。请选择。"
    )
    answer = (
        f"推荐{chosen}。{rejected}售价{over_price}元，大于{budget}元预算，必须排除；"
        f"{chosen}售价{valid_price}元，小于或等于预算。即使{rejected}性能更高也不能选择。"
    )
    return wrap(user, answer, "预算显式比较")


def all_over_budget(rng: random.Random) -> dict:
    a, b = product_names(rng)
    budget = rng.randrange(150, 501, 10)
    a_price = budget + rng.choice([1, 10, 30, 50])
    b_price = budget + rng.choice([5, 20, 40, 80])
    user = (
        f"候选商品只有{a}和{b}，价格分别为{a_price}元、{b_price}元。"
        f"用户最多支付{budget}元，不能追加预算。请问有可推荐商品吗？"
    )
    answer = (
        f"没有合适商品。{a_price}元和{b_price}元都大于{budget}元预算，"
        f"两款均超出预算，因此不能推荐。"
    )
    return wrap(user, answer, "全部超预算")


def all_missing_feature(rng: random.Random) -> dict:
    a, b = product_names(rng)
    feature = rng.choice(FEATURES)
    budget = rng.randrange(300, 701, 10)
    a_price = rng.randrange(100, budget + 1, 10)
    b_price = rng.randrange(100, budget + 1, 10)
    first_negative, second_negative = rng.sample(NEGATIONS, 2)
    user = (
        f"{a}售价{a_price}元，{first_negative.format(feature=feature)}；"
        f"{b}售价{b_price}元，{second_negative.format(feature=feature)}。"
        f"两款都在{budget}元预算内，但用户必须要{feature}。请推荐。"
    )
    answer = (
        f"没有合适商品。虽然两款价格都符合预算，但{a}和{b}均不支持{feature}，"
        "没有任何一款满足必选属性，因此不能推荐。"
    )
    return wrap(user, answer, "全部缺少必选属性")


def partial_insufficient_info(rng: random.Random) -> dict:
    item = f"{rng.choice(PREFIXES)}{rng.randint(100, 999)}款"
    price = rng.randrange(500, 5001, 100)
    known_feature, missing_feature = rng.sample(FEATURES, 2)
    budget = price + rng.randrange(100, 501, 100)
    user = (
        f"{item}售价{price}元，资料确认支持{known_feature}，但没有说明是否支持{missing_feature}。"
        f"用户预算{budget}元，同时必须具备{known_feature}和{missing_feature}。能推荐吗？"
    )
    answer = (
        f"信息不足，暂时无法推荐。价格和{known_feature}已满足，但资料缺少"
        f"{missing_feature}信息，不能判断该硬性条件是否满足。"
    )
    return wrap(user, answer, "单项信息缺失")


GENERATORS: list[Callable[[random.Random], dict]] = [
    budget_priority,
    budget_boundary,
    all_affordable_sort,
    required_feature,
    negative_attribute,
    no_match,
    three_product_sort,
    insufficient_info,
    fact_summary,
]

V3_GENERATORS: list[Callable[[random.Random], dict]] = [
    *GENERATORS,
    explicit_budget_comparison,
    all_over_budget,
    all_missing_feature,
    partial_insufficient_info,
]


def build_split(
    size: int,
    seed: int,
    generators: list[Callable[[random.Random], dict]] = GENERATORS,
) -> list[dict]:
    rng = random.Random(seed)
    rows = [generators[index % len(generators)](rng) for index in range(size)]
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate(rows: list[dict]) -> None:
    seen_prompts: set[str] = set()
    for index, row in enumerate(rows):
        assert row["prompt"][-1]["role"] == "user", index
        assert row["completion"][0]["role"] == "assistant", index
        assert row["completion"][0]["content"].strip(), index
        prompt = row["prompt"][-1]["content"]
        assert prompt not in seen_prompts, f"重复prompt: {index}"
        seen_prompts.add(prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["v2", "v3"], default="v2")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    generators = GENERATORS if args.profile == "v2" else V3_GENERATORS
    train_size = args.train_size or (1800 if args.profile == "v2" else 2600)
    validation_size = args.validation_size or (225 if args.profile == "v2" else 325)
    train = build_split(train_size, args.seed, generators)
    validation = build_split(validation_size, args.seed + 1, generators)
    validate(train)
    validate(validation)

    train_prompts = {row["prompt"][-1]["content"] for row in train}
    validation_prompts = {row["prompt"][-1]["content"] for row in validation}
    if train_prompts & validation_prompts:
        raise ValueError("训练集与验证集存在完全相同的prompt")

    suffix = "" if args.profile == "v2" else "_v3"
    train_path = PROJECT_ROOT / "data" / "processed" / f"sft_train{suffix}.jsonl"
    validation_path = PROJECT_ROOT / "data" / "processed" / f"sft_validation{suffix}.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    print(f"训练集：{len(train)} 条 -> {train_path}")
    print(f"验证集：{len(validation)} 条 -> {validation_path}")
    print("类别：" + "、".join(row["category"] for row in train[:5]))


if __name__ == "__main__":
    main()
