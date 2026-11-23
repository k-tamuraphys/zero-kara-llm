"""エンジン（KV キャッシュ + バッチ）の出力が、素朴な生成と同じか確かめる。"""

import time

import torch

from common import load, make_problems, chat_prompt_ids
from engine import Engine, reply_batch
from llmbook.chat import chat_reply

model, tok = load("sft")
problems = make_problems(1000, seed=777)
questions = [p["question"] for p in problems]

t0 = time.perf_counter()
plain = [chat_reply(model, tok, q) for q in questions]
t_plain = time.perf_counter() - t0

engine = Engine(model, n_slots=64)
t0 = time.perf_counter()
fast = []
for i in range(0, len(questions), 64):
    fast += reply_batch(engine, tok, questions[i:i + 64])
t_fast = time.perf_counter() - t0

same = sum(a == b for a, b in zip(plain, fast))
acc = sum(r == p["target"] for r, p in zip(fast, problems)) / len(problems)
print(f"素朴な生成（1 問ずつ、キャッシュなし）: {t_plain:5.1f} 秒")
print(f"エンジン（キャッシュあり、64 問ずつ）  : {t_fast:5.1f} 秒"
      f"（{t_plain / t_fast:.1f} 倍速い）")
print(f"応答が一致した問題: {same} / {len(problems)}")
print(f"正答率: {acc:.1%}")
