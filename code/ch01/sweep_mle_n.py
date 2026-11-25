"""第 1 章（図の追加）: 投げる回数 n を変えて、最尤推定のばらつきを見る。

表の確率が 0.7 のコインを n 回投げ、表の割合 k/n（最尤推定）を求める。
これを 1 万回くり返し、推定値がどれだけ散らばるかを、n を 5 通りに変えて
比べる。本当の確率 0.7 と、くり返しの回数 1 万は固定する。
    cd code && python ch01/sweep_mle_n.py > ch01/out/sweep_mle_n.txt
出力: ch01/out/sweep_mle_n.csv（ヒストグラム）、sweep_mle_n_q.csv（分位点）
"""
import numpy as np

rng = np.random.default_rng(0)  # 乱数の種を固定する
THETA, TRIALS = 0.7, 10000  # 本当の表の確率、くり返しの回数
NS = [3, 10, 30, 100, 1000]  # 変えるもの: 1 回の実験で投げる回数
# 区間の幅 0.05。中心が 0, 0.05, ..., 1 になるように端を置く
EDGES = np.linspace(-0.025, 1.025, 22)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2

hists, rows = [], []
for n in NS:
    k = rng.binomial(n, THETA, size=TRIALS)  # 各実験で表が出た回数
    est = k / n  # 最尤推定 k/n（1 万個）
    h, _ = np.histogram(est, bins=EDGES)
    hists.append(h / TRIALS)  # 各区間に入った実験の割合
    q05, q50, q95 = np.quantile(est, [0.05, 0.5, 0.95])
    extreme = np.mean((est == 0) | (est == 1))  # 0 か 1 と言い切った割合
    rows.append((n, q05, q50, q95, extreme))
    print(f"n = {n:4d}: 5% 点 {q05:.3f}  中央 {q50:.3f}  95% 点 {q95:.3f}"
          f"  幅 {q95 - q05:.3f}  0 か 1 になった割合 {extreme:.3f}")

with open("ch01/out/sweep_mle_n.csv", "w") as fp:
    fp.write("center," + ",".join(f"n{n}" for n in NS) + "\n")
    for i, c in enumerate(CENTERS):
        vals = ",".join(f"{h[i]:.4f}" for h in hists)
        fp.write(f"{abs(c):.2f},{vals}\n")
with open("ch01/out/sweep_mle_n_q.csv", "w") as fp:
    fp.write("n,q05,q50,q95,extreme\n")
    for r in rows:
        fp.write(f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.3f},{r[4]:.4f}\n")
