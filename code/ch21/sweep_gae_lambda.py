"""GAE の λ を 5 通りに変えて、PPO による倒立振子の学習を比べる。

λ=0 は TD 誤差そのもの、λ=1 はモンテカルロの G_t − V(s_t)。
各 3 個の種、40 反復（640 エピソード）。ほかの設定は ppo.py の既定値。
"""
import time
import torch
from ppo import train_ppo

torch.set_num_threads(2)
OUT = "ch21/out"
t0 = time.time()
lams = [0.0, 0.5, 0.9, 0.95, 1.0]
rows = []
curves = {}
for lam in lams:
    vals, hs = [], []
    for s in range(3):
        h = train_ppo(s, iters=40, lam=lam)[0]
        vals.append(sum(h[-10:]) / 10)          # 最後の 10 反復の平均長
        hs.append(h)
    curves[lam] = [sum(h[i] for h in hs) / 3 for i in range(40)]
    rows.append((lam, sum(vals) / 3, min(vals), max(vals)))
    print(f"λ={lam:4.2f}: 最後の 10 反復の平均長（種ごと）",
          " ".join(f"{v:5.1f}" for v in vals), f" 平均 {sum(vals) / 3:5.1f}")
with open(f"{OUT}/sweep_lambda.csv", "w") as f:
    f.write("idx,lam,mean,lo,hi\n")
    for i, (lam, m, lo, hi) in enumerate(rows):
        f.write(f"{i + 1},{lam},{m:.1f},{lo:.1f},{hi:.1f}\n")
with open(f"{OUT}/sweep_lambda_curves.csv", "w") as f:
    f.write("episodes," + ",".join(f"lam{i}" for i in range(len(lams)))
            + "\n")
    for i in range(40):
        f.write(f"{(i + 1) * 16}," + ",".join(
            f"{curves[l][i]:.1f}" for l in lams) + "\n")
print(f"所要時間 {time.time() - t0:.0f} 秒")
