from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装（应在独立GPU环境安装）"


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA训练前数据与环境体检")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    train_path = PROJECT_ROOT / "data" / "processed" / "sft_train.jsonl"
    validation_path = PROJECT_ROOT / "data" / "processed" / "sft_validation.jsonl"
    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)

    model_is_local = Path(args.model).expanduser().exists()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=model_is_local,
    )
    token_lengths: list[int] = []
    for row in train + validation:
        conversation = row["prompt"] + row["completion"]
        token_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_lengths.append(len(token_ids))

    train_prompts = {row["prompt"][-1]["content"] for row in train}
    validation_prompts = {row["prompt"][-1]["content"] for row in validation}
    overlap = train_prompts & validation_prompts

    report = {
        "packages": {
            name: package_version(name)
            for name in ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes"]
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        },
        "data": {
            "train": len(train),
            "validation": len(validation),
            "train_categories": dict(Counter(row["category"] for row in train)),
            "exact_prompt_overlap": len(overlap),
        },
        "token_lengths": {
            "mean": round(statistics.mean(token_lengths), 2),
            "p50": percentile(token_lengths, 0.5),
            "p95": percentile(token_lengths, 0.95),
            "max": max(token_lengths),
            "over_max_length": sum(length > args.max_length for length in token_lengths),
            "configured_max_length": args.max_length,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if overlap:
        raise RuntimeError("训练集与验证集存在完全相同的prompt，可能发生数据泄漏")
    if report["token_lengths"]["over_max_length"]:
        raise RuntimeError("部分样本超过max_length，将被截断")


if __name__ == "__main__":
    main()

