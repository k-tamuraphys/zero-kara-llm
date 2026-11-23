"""SFT 済みモデルに、学習データにない形（システムプロンプト・2 ターン目）
を与える。

実行:  python ch16/multiturn_test.py [sft|sft_multi]   （既定は sft）
出力:  ch16/out/multiturn_<名前>.txt     CPU で約 2 分。
"""

import random
import sys

import torch

from common import CKPT_DIR, OUT_DIR, device, tok
from llmbook.chat import make_chat_example, render_chat
from llmbook.train import load_checkpoint

ckpt = sys.argv[1] if len(sys.argv) > 1 else "sft"
model = load_checkpoint(CKPT_DIR / f"ch16_{ckpt}.pt", device).eval()
end_id = tok.special_tokens["<|end|>"]


@torch.no_grad()
def reply(messages: list[dict]) -> str:
    """対話の続きとして、アシスタントの次の発話を生成する。"""
    prompt = render_chat(messages, add_generation_prompt=True)
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = model.generate(idx, 16, temperature=0.0, stop_token=end_id)
    return tok.decode(out[0, idx.size(1):].tolist()).split("<|end|>")[0]


# [snippet: multiturn]
system = {"role": "system", "content": "あなたは計算アシスタントです。"}
first = [{"role": "user", "content": "12+35は？"},
         {"role": "assistant", "content": "47です。"}]
second = {"role": "user", "content": "300から123を引くといくつですか？"}
cases = {"1 ターン": [second],
         "システムあり": [system, second],
         "2 ターン目": first + [second]}
# [end: multiturn]

lines = [f"{name}: {reply(m)!r}" for name, m in cases.items()]
# 200 問で正答率を比べる（質問は chat_accuracy と同じ作り方）
rng = random.Random(12345)
n_ok = dict.fromkeys(cases, 0)
for _ in range(200):
    messages, answer = make_chat_example(rng)
    q = messages[0]
    tries = {"1 ターン": [q], "システムあり": [system, q],
             "2 ターン目": first + [q]}
    for name, m in tries.items():
        n_ok[name] += reply(m) == f"{answer}です。"
lines += [f"{name}の正答率: {n / 200:.1%}" for name, n in n_ok.items()]
with open(OUT_DIR / f"multiturn_{ckpt}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
