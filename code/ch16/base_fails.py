"""基盤モデル base.pt に質問し、答えずに「続き」を書くことを確かめる。

実行:  python ch16/base_fails.py
出力:  ch16/out/base_fails.txt, ch16/out/base_next_token.csv   CPU で約 1 分。
"""

import csv
import random
import re

import torch
import torch.nn.functional as F

from common import OUT_DIR, device, load_base, tok
from llmbook.arith import make_problem
from llmbook.chat import render_chat

model = load_base().eval()


# [snippet: continue-text]
@torch.no_grad()
def continue_text(prompt: str, max_new_tokens: int = 16) -> str:
    """prompt の続きを貪欲法で書かせる（特殊トークンも文字として表示）。"""
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = model.generate(idx, max_new_tokens, temperature=0.0)
    return tok.decode(out[0, idx.size(1):].tolist())
# [end: continue-text]


# [snippet: prompts]
chat = render_chat([{"role": "user", "content": "12+35は？"}],
                   add_generation_prompt=True)
prompts = {
    "対話形式": chat,  # <|user|>12+35は？<|end|><|assistant|>
    "地の文で質問": "\n12+35は？",
    "例を見せる": "\n3+4は？7です。\n10+20は？30です。\n12+35は？",
    "ドリル形式": "\n12+35=",
}
# [end: prompts]

lines = []
for name, prompt in prompts.items():
    reply = continue_text(prompt).replace("\n", "⏎")  # 改行を見えるように
    lines.append(f"[{name}] {reply}")

# [snippet: accuracy-by-format]
def leading_number(text: str) -> str:
    """続きの先頭にある数字の並びだけを取り出す（「47です。」→「47」）。"""
    return re.match(r"\d*", text).group()


# 同じ 200 問を 3 通りの書き方でたずね、答えの数字で始まる割合を比べる
rng = random.Random(12345)
n_ok = {"対話形式": 0, "例を見せる": 0, "ドリル形式": 0}
for _ in range(200):
    expr, answer = make_problem(rng)  # 例 "12+35=", "47"
    q = expr.replace("=", "は？")  # 例 "12+35は？"
    tries = {
        "対話形式": render_chat([{"role": "user", "content": q}],
                                add_generation_prompt=True),
        "例を見せる": "\n3+4は？7です。\n10+20は？30です。\n" + q,
        "ドリル形式": "\n" + expr,
    }
    for name, prompt in tries.items():
        n_ok[name] += leading_number(continue_text(prompt, 6)) == answer
# [end: accuracy-by-format]
for name, n in n_ok.items():
    lines.append(f"{name}の正答率: {n / 200:.1%}")


# [snippet: next-token]
@torch.no_grad()
def next_token_top(prompt: str, k: int = 5) -> list[tuple[str, float]]:
    """prompt の次に来るトークンの上位 k 個と確率。"""
    idx = torch.tensor([tok.encode(prompt)], device=device)
    logits, _ = model(idx)
    probs = F.softmax(logits[0, -1], dim=-1)
    top = probs.topk(k)
    return [(tok.decode([i]), p) for p, i in
            zip(top.values.tolist(), top.indices.tolist())]
# [end: next-token]


rows = []
for name in ["対話形式", "地の文で質問"]:
    top = next_token_top(prompts[name])
    shown = ", ".join(f"{t!r} {p:.3f}" for t, p in top)
    lines.append(f"{name}の次トークン: {shown}")
    rows += [{"prompt": name, "rank": r + 1, "token": t, "prob": f"{p:.4f}"}
             for r, (t, p) in enumerate(top)]

with open(OUT_DIR / "base_fails.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
with open(OUT_DIR / "base_next_token.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("\n".join(lines))
