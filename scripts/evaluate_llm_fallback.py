from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="评测真实大模型JSON抽取回退")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/natural-language-recommendations")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/evaluation/llm_fallback_v1.jsonl")
    args = parser.parse_args()
    cases = [json.loads(line) for line in (PROJECT_ROOT / "data/eval/ecommerce_llm_fallback_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            response = client.post(args.endpoint, json={"text": case["text"]})
            payload = response.json()
            selected = payload.get("recommendation", {}).get("selected_product") if response.is_success else None
            source = payload.get("extraction_source") if response.is_success else None
            results.append({
                "id": case["id"],
                "passed": response.is_success and source == "llm_fallback" and selected == case["expected_selected"],
                "status_code": response.status_code,
                "expected_selected": case["expected_selected"],
                "selected_product": selected,
                "extraction_source": source,
                "response": payload,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    summary = {"cases": len(results), "passed": sum(row["passed"] for row in results)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
