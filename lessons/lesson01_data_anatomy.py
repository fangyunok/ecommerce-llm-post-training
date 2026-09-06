"""第 1 课：认识 SFT 样本、聊天模板和训练/推理格式。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sft_demo.jsonl"
ALLOWED_ROLES = {"system", "user", "assistant"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    """逐行读取 JSONL；一行就是一条独立训练样本。"""
    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
    return samples


def validate_sample(sample: dict) -> None:
    """检查最基本的数据约束，尽早发现脏数据。"""
    if not isinstance(sample.get("id"), str) or not sample["id"].strip():
        raise ValueError("样本缺少非空字符串 id")

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{sample['id']}: messages 至少需要两条消息")

    for index, message in enumerate(messages):
        if message.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"{sample['id']}: 第 {index} 条消息 role 非法")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{sample['id']}: 第 {index} 条消息 content 为空")

    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"{sample['id']}: 缺少 user 消息")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{sample['id']}: SFT 样本最后一条必须是 assistant 答案")


def visible_chat_template(messages: list[dict], include_answer: bool) -> str:
    """用于教学的可视化模板；正式训练会改用模型自带 chat template。"""
    selected = messages if include_answer else messages[:-1]
    blocks = [f"<|{item['role']}|>\n{item['content']}\n<|end|>" for item in selected]
    if not include_answer:
        blocks.append("<|assistant|>\n")
    return "\n".join(blocks)


def main() -> None:
    samples = load_jsonl(DATA_PATH)
    for sample in samples:
        validate_sample(sample)

    first = samples[0]
    print(f"数据检查通过：共 {len(samples)} 条样本")
    print(f"第一条样本 ID：{first['id']}")
    print(f"角色顺序：{[message['role'] for message in first['messages']]}")

    print("\n=== SFT 训练时看到的完整文本 ===")
    print(visible_chat_template(first["messages"], include_answer=True))

    print("\n=== 推理时输入模型的提示 ===")
    print(visible_chat_template(first["messages"], include_answer=False))

    print("\n思考题：为什么推理提示不应该提前包含标准答案？")


if __name__ == "__main__":
    main()

