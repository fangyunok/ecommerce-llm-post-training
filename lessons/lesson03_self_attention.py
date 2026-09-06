"""第 3 课：从零实现单头 Self-Attention 与因果掩码。"""

from __future__ import annotations

import math
import sys

import torch
from torch import nn


torch.manual_seed(42)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ToyTokenizer:
    def __init__(self, texts: list[str]) -> None:
        self.id_to_token = sorted(set("".join(texts)))
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.id_to_token)
        }

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[token] for token in text]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)


class SingleHeadSelfAttention(nn.Module):
    """教学版单头自注意力；省略多头、dropout等工程细节。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # 每个 Query 与所有 Key 做点积，得到相关性分数。
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.hidden_dim)

        sequence_length = hidden_states.size(1)
        # 下三角为 True：当前位置只能看自己和左侧历史，不能偷看未来。
        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        context = attention_weights @ values
        return context, attention_weights, causal_mask


def print_matrix(title: str, matrix: torch.Tensor) -> None:
    print(title)
    for row in matrix:
        print("  " + " ".join(f"{value:>6.3f}" for value in row.tolist()))


def main() -> None:
    texts = ["耳机续航", "手机拍照"]
    tokenizer = ToyTokenizer(texts)
    token_ids = torch.tensor([tokenizer.encode(text) for text in texts])

    hidden_dim = 8
    embedding = nn.Embedding(tokenizer.vocab_size, hidden_dim)
    attention = SingleHeadSelfAttention(hidden_dim)
    hidden_states = embedding(token_ids)
    context, weights, causal_mask = attention(hidden_states)

    print("=== 1. 输入与张量形状 ===")
    print(f"两条文本：{texts}")
    print(f"token IDs shape：{tuple(token_ids.shape)}  # [batch, sequence]")
    print(
        f"Embedding shape：{tuple(hidden_states.shape)}  "
        "# [batch, sequence, hidden_dim]"
    )

    print("\n=== 2. 因果掩码（1=允许关注，0=禁止偷看） ===")
    for row in causal_mask.int():
        print("  " + " ".join(str(value) for value in row.tolist()))

    print("\n=== 3. 第一条文本『耳机续航』的注意力权重 ===")
    print("列顺序：耳 机 续 航")
    print_matrix("行顺序：耳/机/续/航", weights[0])

    # 两个“机”的 token embedding 完全相同，但经过注意力后的上下文表示不同。
    machine_position = 1
    raw_difference = torch.norm(
        hidden_states[0, machine_position] - hidden_states[1, machine_position]
    ).item()
    context_difference = torch.norm(
        context[0, machine_position] - context[1, machine_position]
    ).item()

    print("\n=== 4. 同一个『机』在不同上下文中的表示 ===")
    print(f"Attention 前，两个『机』的向量距离：{raw_difference:.6f}")
    print(f"Attention 后，两个『机』的向量距离：{context_difference:.6f}")
    print("解释：前者只是同一个 token 的查表结果；后者融合了『耳』或『手』的信息。")

    print("\n注意：本课参数尚未训练，权重数值没有业务含义；我们验证的是计算机制。")


if __name__ == "__main__":
    main()

