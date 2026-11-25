"""3 択バンディットの勾配推定を、乱数の種を変えて何本も走らせる。

bandit_pg.py の図（bandit_running.csv）は 1 本のサンプル列だった。
ここでは種を 8 通りに変え、サンプル数 n とともに推定の揺れの幅が
どう縮むかを見る。bandit_pg.py は読み込まず（実行すると出力を
上書きするため）、同じ推定式をここに書く。
"""
import torch

OUT = "ch21/out"
r = torch.tensor([1.0, 0.0, 0.0])           # 正解の a_1 だけ報酬 1
theta = torch.zeros(3)                      # θ=0: 3 つとも確率 1/3


def score_estimates(theta, r, n):
    """n 個のサンプルそれぞれの推定 r(a)∇log π(a) を返す（第 1 成分）。"""
    pi = torch.softmax(theta, dim=0)
    a = torch.multinomial(pi, n, replacement=True)
    score = torch.nn.functional.one_hot(a, 3).float() - pi
    return (r[a][:, None] * score)[:, 0]     # (n,)


seeds = range(8)
grid = sorted({int(round(10 ** (k / 20))) for k in range(0, 81)})
runs = []
for s in seeds:
    torch.manual_seed(100 + s)               # bandit_pg.py の種と重ねない
    g = score_estimates(theta, r, 10000)
    runs.append(g.cumsum(0) / torch.arange(1, 10001))   # 最初の n 個の平均
with open(f"{OUT}/sweep_bandit_seeds.csv", "w") as f:
    f.write("n," + ",".join(f"s{s}" for s in seeds) + ",lo,hi,sd\n")
    for n in grid:
        v = [run[n - 1].item() for run in runs]
        sd = (8 / 81 / n) ** 0.5     # 理論の標準偏差（1 個の分散は 8/81）
        f.write(f"{n}," + ",".join(f"{x:.5f}" for x in v)
                + f",{min(v):.5f},{max(v):.5f},{sd:.5f}\n")
for n in [10, 100, 1000, 10000]:
    v = [run[n - 1].item() for run in runs]
    print(f"n={n:>5}: 8 本の推定の最小 {min(v):.3f}  最大 {max(v):.3f}"
          f"  幅 {max(v) - min(v):.3f}")
