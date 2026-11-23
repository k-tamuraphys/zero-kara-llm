"""ハイパーパラメータ感度: クリップの有無と学習率を変えて学習を比べる。"""
import time
import torch
from reinforce import train, summarize
from ppo import train_ppo

torch.set_num_threads(2)
OUT = "ch21/out"
t0 = time.time()

# (1) クリップを外す（ε を事実上 ∞ にする）と何が起きるか
runs = [train_ppo(s, clip=1e9)[0] for s in range(5)]
last = [sum(r[-10:]) / 10 for r in runs]
print("クリップなし: 最後の 10 反復の平均エピソード長（種ごと）",
      " ".join(f"{x:5.1f}" for x in last), f" 平均 {sum(last)/5:5.1f}")
with open(f"{OUT}/noclip_curves.csv", "w") as f:
    f.write("episodes,noclip_mean,noclip_lo,noclip_hi\n")
    for i, (m, lo, hi) in enumerate(summarize(runs)):
        f.write(f"{(i + 1) * 16},{m:.1f},{lo:.1f},{hi:.1f}\n")
# 種ごとの曲線も保存（崩れ方を 1 本ずつ見るため）
with open(f"{OUT}/noclip_seeds.csv", "w") as f:
    f.write("episodes," + ",".join(f"s{k}" for k in range(5)) + "\n")
    for i in range(len(runs[0])):
        f.write(f"{(i + 1) * 16}," + ",".join(f"{r[i]:.1f}" for r in runs)
                + "\n")

# (2) 学習率: 40 反復（640 エピソード）の後半 10 反復の平均で比べる
print("\n学習率  REINFORCE+ベースライン   PPO   （3 つの種の平均）")
with open(f"{OUT}/lr_sweep.csv", "w") as f:
    f.write("lr,reinforce,reinforce_lo,reinforce_hi,ppo,ppo_lo,ppo_hi\n")
    for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
        res = []
        for algo in ["reinforce", "ppo"]:
            vals = []
            for s in range(3):
                if algo == "reinforce":
                    h = train("baseline", s, iters=40, lr=lr)[0]
                else:
                    h = train_ppo(s, iters=40, lr=lr)[0]
                vals.append(sum(h[-10:]) / 10)
            res.append(vals)
        print(f"{lr:6.0e}  " + "   ".join(
            "/".join(f"{v:3.0f}" for v in vals) for vals in res))
        f.write(f"{lr}," + ",".join(
            f"{sum(v)/3:.1f},{min(v):.1f},{max(v):.1f}" for v in res) + "\n")
print(f"所要時間 {time.time()-t0:.0f} 秒")
