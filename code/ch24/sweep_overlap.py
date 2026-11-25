"""学習中に出た問題と、評価の問題の重なりを数える（学習の長さを変えて）。
train_agent_rl.py と同じ種で学習の問題の列を作り直すだけ（モデルは使わない）。
  python ch24/sweep_overlap.py > ch24/out/sweep_overlap.txt
"""
import random
import re

from env import *

Q = 16
KINDS = ["add", "add", "mul", "two"]
ev = eval_sets()["add"]
ev_q = {t.question for t in ev}                    # 評価の加減算の質問文
rng = random.Random(2024)                          # 学習と同じ種
seen, n_add = set(), 0
marks = {25, 50, 73, 91, 103}
print("ステップ数  学習の加減算の問題数  評価の 1,000 問のうち学習に出た問題")
for step in range(1, max(marks) + 1):
    tasks = [make_task(rng, rng.choice(KINDS)) for _ in range(Q)]
    for t in tasks:
        if t.kind == "add":
            n_add += 1
            if t.question in ev_q:
                seen.add(t.question)
    if step in marks:
        hit = [t for t in ev if t.question in seen]
        print(f"{step:8d}  {n_add:16d}  {len(hit):4d} 問"
              f"（{len(hit) / len(ev):.1%}）")
hit = [t for t in ev if t.question in seen]
small = sum(max(map(int, re.findall(r"\d+", t.question)[:2])) < 10
            for t in hit) if hit else 0
print(f"103 ステップまでに重なった問題のうち、両方の数が 1 桁: {small} 問")
for t in hit[:5]:
    print("  例:", t.question)
