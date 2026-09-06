from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data/eval/ecommerce_robustness_v1.jsonl"


def main() -> None:
    cases: list[dict] = []
    names = [
        ("星河", "云帆", "青岚"),
        ("远山", "流光", "墨羽"),
        ("晨星", "北辰", "南风"),
        ("轻舟", "飞鸿", "月白"),
        ("松影", "竹语", "海棠"),
    ]

    for index, (a, b, c) in enumerate(names, 1):
        cases.append({
            "id": f"free_name_price_{index:02d}",
            "category": "自由名称",
            "text": f"{a}台灯售价{99+index}元；{b}台灯售价{89+index}元；{c}台灯售价{109+index}元。三款功能相同，选择价格最低的。",
            "expected_selected": f"{b}台灯",
        })
        cases.append({
            "id": f"unit_weight_{index:02d}",
            "category": "单位换算",
            "text": f"{a}电脑价格{4800+index}元、重{1300+index*10}g；{b}电脑价格{5000+index}元、重1.2kg；{c}电脑价格{4700+index}元、重1.6kg。预算不超过{4900+index}元，重量不高于1.4kg，优先低价。",
            "expected_selected": f"{a}电脑",
        })
        cases.append({
            "id": f"capacity_filter_{index:02d}",
            "category": "阈值过滤",
            "text": f"{a}充电宝价格{79+index}元、容量8000mAh；{b}充电宝价格{109+index}元、容量12000mAh；{c}充电宝价格{99+index}元、容量12000mAh。容量不低于10000mAh，选择最低价。",
            "expected_selected": f"{c}充电宝",
        })
        cases.append({
            "id": f"multi_sort_{index:02d}",
            "category": "多级排序",
            "text": f"{a}移动盘售价{299+index}元、容量256GB；{b}移动盘售价{259+index}元、容量256GB；{c}移动盘售价{199+index}元、容量128GB。优先容量最大，容量相同选价格最低。",
            "expected_selected": f"{b}移动盘",
        })
        cases.append({
            "id": f"no_match_{index:02d}",
            "category": "无合适商品",
            "text": f"{a}耳机售价{399+index}元、续航30小时；{b}耳机售价{499+index}元、续航50小时。预算不超过300元，优先续航。",
            "expected_selected": None,
            "expected_outcome": "no_match",
        })
        cases.append({
            "id": f"missing_field_{index:02d}",
            "category": "字段缺失",
            "text": f"{a}电脑售价{4500+index}元；{b}电脑售价{4700+index}元。预算不超过5000元，重量不高于1.3kg，优先低价。",
            "expected_selected": None,
            "expected_outcome": "insufficient",
        })
        cases.append({
            "id": f"conflicting_constraints_{index:02d}",
            "category": "冲突条件",
            "text": f"{a}存储卡售价{99+index}元、容量64GB；{b}存储卡售价{129+index}元、容量128GB。容量至少128GB且不得超过64GB，选择最低价。",
            "expected_selected": None,
            "expected_outcome": "no_match",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(f"generated={len(cases)} output={OUTPUT}")


if __name__ == "__main__":
    main()
