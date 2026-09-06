"""第 2 课：用一个极简字符级语言模型理解 next-token prediction。"""

from __future__ import annotations

import random
import sys

import torch
from torch import nn


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 为了让 token 拆分肉眼可见，本课把“每个字符”当作一个 token。
# 真实大模型使用训练好的 tokenizer，一个 token 不一定等于一个字符。
CORPUS = (
    "耳机续航很长。"
    "耳机音质很好。"
    "手机续航很长。"
    "手机拍照很好。"
) * 80


class CharTokenizer:
    """只服务于本课的最小字符级 tokenizer。"""

    def __init__(self, text: str) -> None:
        self.id_to_token = sorted(set(text))
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.id_to_token)
        }

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[token] for token in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)


class TinyNextTokenModel(nn.Module):
    """根据当前 token 预测下一个 token 的极简模型。"""

    def __init__(self, vocab_size: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        logits = self.output(embeddings)
        return logits


def make_training_pairs(token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """错开一位：当前位置是输入，下一位置是监督标签。"""
    inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
    labels = torch.tensor(token_ids[1:], dtype=torch.long)
    return inputs, labels


def show_top_predictions(
    model: TinyNextTokenModel,
    tokenizer: CharTokenizer,
    current_token: str,
    top_k: int = 5,
) -> None:
    token_id = torch.tensor([tokenizer.token_to_id[current_token]])
    with torch.no_grad():
        logits = model(token_id)[0]
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=min(top_k, tokenizer.vocab_size))

    predictions = [
        f"{tokenizer.id_to_token[index]}={probability:.3f}"
        for probability, index in zip(top_probs.tolist(), top_ids.tolist())
    ]
    print(f"看到 token『{current_token}』后，下一个 token 的候选：{', '.join(predictions)}")


def main() -> None:
    tokenizer = CharTokenizer(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    inputs, labels = make_training_pairs(token_ids)

    print("=== 1. Tokenization ===")
    example = "耳机续航"
    example_ids = tokenizer.encode(example)
    print(f"原文：{example}")
    print(f"tokens：{list(example)}")
    print(f"token IDs：{example_ids}")
    print(f"词表大小：{tokenizer.vocab_size}")

    print("\n=== 2. 训练样本如何错开一位 ===")
    print(f"输入前 8 个 token：{tokenizer.decode(inputs[:8].tolist())}")
    print(f"标签前 8 个 token：{tokenizer.decode(labels[:8].tolist())}")

    model = TinyNextTokenModel(tokenizer.vocab_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    print("\n=== 3. 训练前的随机预测 ===")
    show_top_predictions(model, tokenizer, "续")

    print("\n=== 4. 开始训练 ===")
    for step in range(201):
        logits = model(inputs)
        loss = loss_function(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step={step:>3}, loss={loss.item():.4f}")

    print("\n=== 5. 训练后的预测 ===")
    show_top_predictions(model, tokenizer, "续")
    show_top_predictions(model, tokenizer, "航")

    embedding = model.embedding(
        torch.tensor([tokenizer.token_to_id["耳"]])
    )
    print("\n=== 6. Embedding ===")
    print(f"token『耳』的 ID：{tokenizer.token_to_id['耳']}")
    print(f"embedding shape：{tuple(embedding.shape)}")
    print(f"embedding 前 5 个数：{embedding[0, :5].detach().tolist()}")


if __name__ == "__main__":
    main()

