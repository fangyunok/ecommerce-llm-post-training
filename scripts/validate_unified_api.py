from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="验证统一购物助手的三条路由")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/shopping-assistant")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/evaluation/unified_api_v1.json")
    args = parser.parse_args()
    cases = [
        {"id": "rules", "request": {"text": "A款100元；B款90元。选择价格最低的。"}, "route": "deterministic_rules", "selected": "B款"},
        {"id": "fallback", "request": {"text": "移动电源有轻羽版，容量10000mAh价格99元；还有长风版，容量20000mAh价格169元。需求是容量至少15000mAh，价格低者优先。"}, "route": "llm_extraction_rules", "selected": "长风版"},
        {"id": "chat", "request": {"text": "用一句话解释什么是预算。", "mode": "chat"}, "route": "language_model", "selected": None},
    ]
    results = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            response = client.post(args.endpoint, json=case["request"])
            payload = response.json()
            selected = (payload.get("recommendation") or {}).get("selected_product")
            passed = response.is_success and payload.get("route") == case["route"]
            if case["selected"] is not None:
                passed = passed and selected == case["selected"]
            results.append({"id": case["id"], "passed": passed, "status_code": response.status_code, "response": payload})
    artifact = {"cases": len(results), "passed": sum(row["passed"] for row in results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": artifact["cases"], "passed": artifact["passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
