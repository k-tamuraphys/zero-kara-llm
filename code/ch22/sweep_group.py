"""グループの大きさ G を変えると、学びどきのグループの割合はどう変わるか。
sft.pt に、訓練と同じ作り方の質問 200 問を 32 本ずつ答えさせ（温度 1）、
32 本を G 本ずつのグループに分けて、正解と不正解が混ざる割合を数える。
あわせて、学習の 1 周の図に使う「16 問 × 8 本」の採点結果を書き出す。
  python ch22/sweep_group.py > ch22/out/sweep_group.txt
"""
import csv
import random
import time

import torch

from common import (OUT_DIR, load_sft, prompt_ids, rollout, reward,
                    make_chat_example)

torch.manual_seed(0)
rng = random.Random(0)  # rl_arith.py と同じ作り方・同じ種の訓練用の問題
N, M = 200, 32           # 質問の数、1 問あたりのサンプル数
Gs = [2, 4, 8, 16, 32]

policy = load_sft()
t0 = time.time()
R = []                   # R[q][j]: 質問 q の j 本目の報酬（0 か 1）
for _ in range(N):
    msgs, ans = make_chat_example(rng)
    p = prompt_ids(msgs[0]["content"])
    resp, _ = rollout(policy, p, M)
    R.append([reward(o.tolist(), ans) for o in resp])
R = torch.tensor(R)      # (N, M)

# G 本ずつに区切り、正解と不正解が混ざるグループの割合を数える
rows = []
for G in Gs:
    g = R.view(N, M // G, G)                 # (N, M/G, G)
    s = g.sum(dim=2)
    mixed = ((s > 0) & (s < G)).float()      # 学びどきなら 1
    rows.append({"G": G, "signal": round(mixed.mean().item(), 4),
                 "all1": round((s == G).float().mean().item(), 4),
                 "all0": round((s == 0).float().mean().item(), 4),
                 "episodes": 16 * G})        # Q=16 問のときの本数
with open(OUT_DIR / "sweep_group.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# 質問ごとの正答率（32 本中の正解の割合）の分布
p_hat = R.float().mean(dim=1)
with open(OUT_DIR / "sweep_group_phat.csv", "w", newline="") as f:
    f.write("lo,count\n")
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        n = ((p_hat >= lo) & ((p_hat < hi) | (k == 9))).sum().item()
        f.write(f"{lo:.1f},{n}\n")

# 学習の 1 周の図：最初の 16 問 × 8 本（1 ステップぶん）の採点結果
# c は図の色分け：1 = 学びどきの正解（A>0）、0 = 学びどきの不正解（A<0）、
# 2 = 全員が同じ報酬のグループ（A=0）
step = R[:16, :8]
ss = step.sum(dim=1)
with open(OUT_DIR / "sweep_group_step.csv", "w", newline="") as f:
    f.write("q,i,r,c\n")
    for q in range(16):
        same = ss[q] == 0 or ss[q] == 8
        for i in range(8):
            c = 2 if same else int(R[q, i])
            f.write(f"{q + 1},{i + 1},{int(R[q, i])},{c}\n")
print(f"questions {N}, samples per question {M}  ({time.time()-t0:.0f}s)")
print(f"mean reward (temperature 1) {R.float().mean():.3f}")
for r in rows:
    print(f"G={r['G']:2d}  signal {r['signal']:.3f}  all1 {r['all1']:.3f}"
          f"  all0 {r['all0']:.3f}  episodes/step {r['episodes']}")
print(f"step example: correct {int(step.sum())}/128, "
      f"signal groups {int(((ss > 0) & (ss < 8)).sum())}/16")
