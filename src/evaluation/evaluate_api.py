from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "ecommerce_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "baseline.jsonl"


@dataclass
class CheckResult:
    passed: bool
    passed_checks: list[str]
    failed_checks: list[str]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalized(text: str) -> str:
    return "".join(text.lower().split())


def check_answer(answer: str, checks: dict) -> CheckResult:
    text = normalized(answer)
    passed_checks: list[str] = []
    failed_checks: list[str] = []

    for phrase in checks.get("all_of", []):
        label = f"包含:{phrase}"
        (passed_checks if normalized(phrase) in text else failed_checks).append(label)

    any_of = checks.get("any_of", [])
    if any_of:
        label = "至少包含其一:" + "|".join(any_of)
        target = passed_checks if any(normalized(item) in text for item in any_of) else failed_checks
        target.append(label)

    for phrase in checks.get("none_of", []):
        label = f"不得包含:{phrase}"
        (passed_checks if normalized(phrase) not in text else failed_checks).append(label)

    return CheckResult(
        passed=not failed_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评测已部署的大模型API")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(cases, start=1):
            response = client.post(
                args.endpoint,
                json={
                    "messages": case["messages"],
                    "max_new_tokens": 160,
                    "do_sample": False,
                    "repetition_penalty": 1.05,
                },
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload["message"]["content"]
            check = check_answer(answer, case["checks"])

            result = {
                "id": case["id"],
                "category": case["category"],
                "passed": check.passed,
                "passed_checks": check.passed_checks,
                "failed_checks": check.failed_checks,
                "answer": answer,
                "usage": payload["usage"],
                "latency_seconds": payload["latency_seconds"],
            }
            results.append(result)
            symbol = "PASS" if check.passed else "FAIL"
            print(f"[{index}/{len(cases)}] {symbol} {case['id']}：{answer}")

    with args.output.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    passed = sum(result["passed"] for result in results)
    latencies = [result["latency_seconds"] for result in results]
    total_completion_tokens = sum(
        result["usage"]["completion_tokens"] for result in results
    )
    try:
        display_result_file = str(args.output.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        display_result_file = str(args.output)
    summary = {
        "cases": len(results),
        "passed": passed,
        "strict_accuracy": round(passed / len(results), 4),
        "mean_latency_seconds": round(statistics.mean(latencies), 3),
        "completion_tokens": total_completion_tokens,
        "result_file": display_result_file,
    }
    summary_path = args.output.with_suffix(".summary.json")
    try:
        summary["summary_file"] = str(summary_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        summary["summary_file"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

