from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ecommerce_llm.recommendation_engine import recommend_by_rules
from src.ecommerce_llm.schemas import RuleRecommendationRequest
from src.evaluation.evaluate_api import check_answer


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="合并规则引擎与已有模型输出并评测")
    parser.add_argument(
        "--model-results",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/sft_1_5b_v2_extended.jsonl",
    )
    parser.add_argument(
        "--rule-cases",
        type=Path,
        default=PROJECT_ROOT / "data/eval/ecommerce_rules_eval.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/evaluation/hybrid_rules_v1_extended.jsonl",
    )
    args = parser.parse_args()

    results = {row["id"]: row for row in read_jsonl(args.model_results)}
    rule_ids: list[str] = []
    for case in read_jsonl(args.rule_cases):
        case_id = case["id"]
        if case_id not in results:
            raise ValueError(f"规则题目不在模型结果中：{case_id}")
        request = RuleRecommendationRequest.model_validate(case["request"])
        response = recommend_by_rules(request)
        checked = check_answer(response.answer, case["checks"])
        results[case_id] = {
            "id": case_id,
            "category": case["category"],
            "passed": checked.passed,
            "passed_checks": checked.passed_checks,
            "failed_checks": checked.failed_checks,
            "answer": response.answer,
            "decision_source": response.decision_source,
            "latency_seconds": 0.0,
        }
        rule_ids.append(case_id)

    ordered = list(results.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in ordered:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "passed": 0})
    for row in ordered:
        categories[row["category"]]["cases"] += 1
        categories[row["category"]]["passed"] += int(row["passed"])
    summary = {
        "cases": len(ordered),
        "passed": sum(row["passed"] for row in ordered),
        "strict_accuracy": round(sum(row["passed"] for row in ordered) / len(ordered), 4),
        "rule_routed_cases": len(rule_ids),
        "rule_routed_passed": sum(results[case_id]["passed"] for case_id in rule_ids),
        "by_category": {
            category: {
                **stats,
                "strict_accuracy": round(stats["passed"] / stats["cases"], 4),
            }
            for category, stats in sorted(categories.items())
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
