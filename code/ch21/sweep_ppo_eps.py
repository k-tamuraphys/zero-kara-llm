"""PPO のクリップの幅 ε を 5 通りに変えて、倒立振子の学習を比べる。

ε=∞ は sensitivity.py と同じく 1e9（事実上クリップなし）で表す。
各 3 個の種、40 反復（640 エピソード）。ほかの設定は ppo.py の既定値。
"""
import time
import torch
from ppo import train_ppo

torch.set_num_threads(2)
OUT = "ch21/out"
t0 = time.time()
eps_list = [0.05, 0.1, 0.2, 0.5, 1e9]
names = ["0.05", "0.1", "0.2", "0.5", "inf"]
last, kl = {}, {}
for e, name in zip(eps_list, names):
    vals, kls = [], []
    for s in range(3):
        log = []
        h = train_ppo(s, iters=40, clip=e, log=log)[0]
        vals.append(sum(h[-10:]) / 10)          # 最後の 10 反復の平均長
        kls.append([row[2] for row in log])     # 反復ごとの KL の推定
    last[name] = vals
    kl[name] = [sum(k[i] for k in kls) / 3 for i in range(40)]
    print(f"ε={name:>4}: 最後の 10 反復の平均長（種ごと）",
          " ".join(f"{v:5.1f}" for v in vals),
          f" KL の平均 {sum(kl[name]) / 40:.4f}"
          f"  KL の最大 {max(max(k) for k in kls):.4f}")
with open(f"{OUT}/sweep_eps.csv", "w") as f:
    f.write("idx,eps,mean,lo,hi\n")
    for i, name in enumerate(names):
        v = last[name]
        f.write(f"{i + 1},{name},{sum(v) / 3:.1f},{min(v):.1f},{max(v):.1f}\n")
with open(f"{OUT}/sweep_eps_kl.csv", "w") as f:
    f.write("iter," + ",".join(f"kl_{n}" for n in names) + "\n")
    for i in range(40):
        f.write(f"{i + 1}," + ",".join(f"{kl[n][i]:.5f}" for n in names)
                + "\n")
print(f"所要時間 {time.time() - t0:.0f} 秒")
