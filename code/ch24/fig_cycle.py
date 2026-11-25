"""学習の 1 周の図（第 24 章「学習の 1 周」）のためのデータを作る。
SFT+RL の学習の 1 ステップ目と同じ乱数で、コールドスタート後のモデル
ch24_cold.pt に 16 問 × 8 本をロールアウトさせ、軌跡ごとの電卓の回数・
報酬（c=0.2）・アドバンテージを書き出す。学習（更新）はしない。
  python ch24/fig_cycle.py > ch24/out/fig_cycle.txt  → out/fig_cycle.csv
"""
import csv
import random
import time

import torch
from env import *

torch.set_num_threads(2)
t0 = time.time()
Q, G, C = 16, 8, 0.2
# train_agent_rl.py の 1 ステップ目と同じ種と同じ手順で問題と軌跡を作る
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
rng = random.Random(2024)
KINDS = ["add", "add", "mul", "two"]
model = load_checkpoint(CKPT_DIR / "ch24_cold.pt")
tasks = [make_task(rng, rng.choice(KINDS)) for _ in range(Q)]
group = [t for t in tasks for _ in range(G)]               # 1 問に G 本
trajs = rollout(model, group, temperature=1.0, eps=0.2, gen=gen)
r = torch.tensor([reward(t, C) for t in trajs]).view(Q, G)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)

with open(OUT_DIR / "fig_cycle.csv", "w", newline="") as f:
    w = csv.writer(f)
    # i: グループの中の番号、q: 問題の番号（図の行）
    w.writerow(["i", "q", "kind", "calls", "correct", "reward", "adv",
                "rcls", "acls", "ntok"])
    for k, t in enumerate(trajs):
        qi, gi = divmod(k, G)
        a = adv[qi, gi].item()
        rcls = {1.0: 3, 0.8: 2, 0.0: 1}.get(round(r[qi, gi].item(), 3), 0)
        # 正・負・0。全員同じ報酬の行は、丸めの誤差で ±0.001 程度になる
        acls = 1 if a > 0.01 else (0 if a < -0.01 else 2)
        w.writerow([gi + 1, qi + 1, t.task.kind, t.n_calls,
                    int(reward(t, 0) == 1.0), round(r[qi, gi].item(), 3),
                    round(a, 3), rcls, acls, sum(t.mask)])

print(f"問題 {Q} 問 × {G} 本 = {Q * G} 本（c={C}、温度 1、eps=0.2）")
for q in range(Q):
    row = trajs[q * G:(q + 1) * G]
    print(f"  {q + 1:2d} {tasks[q].kind}  {tasks[q].question:24s}"
          f" 電卓 {sum(t.n_calls > 0 for t in row)}/{G}"
          f"  報酬 {' '.join(f'{x:+.1f}' for x in r[q].tolist())}")
same = (r.std(1) == 0).sum().item()
print(f"報酬の平均 {r.mean():.4f}、全員が同じ報酬のグループ {same} / {Q}")
turns = [sum(not x.startswith("<|result|>") for x in t.texts)
         for t in trajs]                          # モデルの発話の数
print("ターン数の内訳:", {k: turns.count(k) for k in sorted(set(turns))})
ntok = [sum(t.mask) for t in trajs]
print(f"モデルが書いたトークン数: 最小 {min(ntok)}、最大 {max(ntok)}、"
      f"合計 {sum(ntok)}")
print(f"[{time.time() - t0:.0f} 秒]")
