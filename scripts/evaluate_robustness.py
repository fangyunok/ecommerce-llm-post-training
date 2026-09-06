from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ecommerce_llm.constraint_extractor import extract_rule_request
from src.ecommerce_llm.recommendation_engine import recommend_by_rules


def main() -> None:
    source = PROJECT_ROOT / "data/eval/ecommerce_robustness_v1.jsonl"
    cases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    results = []
    category_stats = defaultdict(lambda: {"cases": 0, "passed": 0})
    for case in cases:
        try:
            extracted = extract_rule_request(case["text"])
            response = recommend_by_rules(extracted)
            if case.get("expected_outcome") == "no_match":
                passed = response.selected_product is None and "没有合适商品" in response.answer
            elif case.get("expected_outcome") == "insufficient":
                passed = response.selected_product is None and "信息不足" in response.answer
            else:
                passed = response.selected_product == case["expected_selected"]
            error = None
        except Exception as exc:
            response = None
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        category_stats[case["category"]]["cases"] += 1
        category_stats[case["category"]]["passed"] += int(passed)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "passed": passed,
            "expected_selected": case.get("expected_selected"),
            "selected_product": response.selected_product if response else None,
            "answer": response.answer if response else None,
            "error": error,
        })

    output = PROJECT_ROOT / "outputs/evaluation/robustness_v1.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = {
        "cases": len(results),
        "passed": sum(row["passed"] for row in results),
        "strict_accuracy": round(sum(row["passed"] for row in results) / len(results), 4),
        "by_category": {
            category: {
                **stats,
                "strict_accuracy": round(stats["passed"] / stats["cases"], 4),
            }
            for category, stats in sorted(category_stats.items())
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
