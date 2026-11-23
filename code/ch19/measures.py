"""三つの物差し: 損失・パープレキシティ・トークン単位の正解率・完全一致。"""

import math
import random

import torch
import torch.nn.functional as F

from common import load, evaluate
from llmbook.chat import encode_chat_example, make_chat_example

model, tok = load("sft")
rng = random.Random(777)
total_loss, n_tokens, n_token_ok, n_all_ok, N = 0.0, 0, 0, 0, 1000
for _ in range(N):
    messages, _ = make_chat_example(rng)
    ids, labels = encode_chat_example(tok, messages)
    # [snippet: token-loss]
    x, y = torch.tensor([ids]), torch.tensor(labels)
    with torch.no_grad():
        logits, _ = model(x)                      # (1, T, V)
    mask = y != -100                              # 応答の部分だけを採点する
    loss = F.cross_entropy(logits[0][mask], y[mask], reduction="sum")
    ok = logits[0][mask].argmax(dim=-1) == y[mask]  # 各トークンの当たり外れ
    # [end: token-loss]
    total_loss += loss.item()
    n_tokens += int(mask.sum())
    n_token_ok += int(ok.sum())
    n_all_ok += int(ok.all())

mean_loss = total_loss / n_tokens
exact = sum(p["correct"] for p in evaluate("sft", N, 777)) / N
print(f"応答 1 件あたりのトークン数      : {n_tokens / N:.2f}")
print(f"1 トークンあたりの損失           : {mean_loss:.4f}")
print(f"パープレキシティ                 : {math.exp(mean_loss):.3f}")
print(f"トークン単位の正解率（お手本つき）: {n_token_ok / n_tokens:.1%}")
print(f"全トークン正解の応答（お手本つき）: {n_all_ok / N:.1%}")
print(f"完全一致の正答率（自力で生成）    : {exact:.1%}")
