from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ecommerce_llm.constraint_extractor import extract_rule_request
from src.ecommerce_llm.recommendation_engine import recommend_by_rules
from src.ecommerce_llm.schemas import RuleRecommendationRequest
from src.evaluation.evaluate_api import check_answer


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def signature(request: RuleRecommendationRequest) -> dict:
    def canonical_field(field: str, value: float) -> str:
        if field == "capacity" and value >= 1000:
            return "battery_capacity"
        if field == "rate":
            return "refresh_rate"
        return field

    return {
        "products": [
            {
                "name": product.name,
                "values": {
                    canonical_field(field, value): value
                    for field, value in product.values.items()
                },
            }
            for product in request.products
        ],
        "constraints": [
            {
                **constraint.model_dump(),
                "field": canonical_field(constraint.field, constraint.value),
            }
            for constraint in request.constraints
        ],
        "sort": request.sort.model_dump(),
    }


def evaluate_case(case_id: str, text: str, gold: dict, checks: dict, split: str) -> dict:
    extracted = extract_rule_request(text)
    response = recommend_by_rules(extracted)
    checked = check_answer(response.answer, checks)
    exact = signature(extracted) == signature(RuleRecommendationRequest.model_validate(gold))
    return {
        "id": case_id,
        "split": split,
        "extraction_exact": exact,
        "decision_passed": checked.passed,
        "answer": response.answer,
        "extracted": signature(extracted),
    }


def main() -> None:
    natural_cases = {
        row["id"]: row
        for row in read_jsonl(PROJECT_ROOT / "data/eval/ecommerce_eval_v2.jsonl")
    }
    original_gold = read_jsonl(PROJECT_ROOT / "data/eval/ecommerce_rules_eval.jsonl")
    results = []
    for row in original_gold:
        natural = natural_cases[row["id"]]
        results.append(
            evaluate_case(
                row["id"],
                natural["messages"][0]["content"],
                row["request"],
                natural["checks"],
                "original",
            )
        )
    for row in read_jsonl(PROJECT_ROOT / "data/eval/ecommerce_extraction_holdout.jsonl"):
        results.append(
            evaluate_case(
                row["id"], row["text"], row["gold_request"], row["checks"], "holdout"
            )
        )

    output = PROJECT_ROOT / "outputs/evaluation/extraction_v1.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = {
        "cases": len(results),
        "extraction_exact": sum(row["extraction_exact"] for row in results),
        "decision_passed": sum(row["decision_passed"] for row in results),
        "by_split": {
            split: {
                "cases": sum(row["split"] == split for row in results),
                "extraction_exact": sum(
                    row["split"] == split and row["extraction_exact"] for row in results
                ),
                "decision_passed": sum(
                    row["split"] == split and row["decision_passed"] for row in results
                ),
            }
            for split in ("original", "holdout")
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
