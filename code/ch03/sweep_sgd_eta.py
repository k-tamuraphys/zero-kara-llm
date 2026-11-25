"""ミニバッチ SGD の進み方と揺れが、B と学習率でどう変わるか調べる。

sgd_minibatch.py の実験 2 と同じデータ（1 万件）と同じ手順（エポックごとに
順番を混ぜ、B 件ずつ使う）で、B を 3 通り、学習率を 5 通りに変える。
(1) (w, b) = (0, 0) から 3 エポック（3 万件）。3 エポック目の毎ステップの損失
    （1 万件で評価）の平均が、最小値をどれだけ上回るか。進む速さと揺れの両方。
(2) 谷底 (w*, b*) から 1 エポック。毎ステップの損失の平均が最小値を
    どれだけ上回るか。谷底にいても止まらずに離れてしまう分、つまり揺れだけ。
どちらも乱数の種を 3 つ変えて走らせ、中央値と最小・最大を記録する。
"""

from pathlib import Path

import numpy as np

from linreg_numpy import grad, mse

OUT = Path(__file__).parent / "out"
data = np.loadtxt(OUT / "char_token_big.csv", delimiter=",", skiprows=1)
x, y = data[:, 0] / 100, data[:, 1] / 100
N = len(x)  # 10000
X1 = np.stack([x, np.ones(N)], 1)  # 損失を速く計算するための (N, 2)
best = np.linalg.lstsq(X1, y, rcond=None)[0]
L_min = mse(best[0], best[1], x, y)  # 到達できる最小の損失


def run(eta, B, seed, start, n_epochs):
    """SGD を n_epochs 回まわし、最後のエポックの毎ステップの損失を返す。"""
    rng = np.random.default_rng(seed)
    w, b = start
    losses = []
    for epoch in range(n_epochs):
        perm = rng.permutation(N)  # エポックごとに順番を混ぜる
        for s in range(0, N, B):
            idx = perm[s:s + B]
            dw, db = grad(w, b, x[idx], y[idx])
            w, b = w - eta * dw, b - eta * db
            if epoch == n_epochs - 1:
                losses.append(np.mean((X1 @ (w, b) - y) ** 2))
    return np.array(losses)


print(f"到達できる最小の損失: {L_min:.5f}")
etas = [0.003, 0.01, 0.03, 0.1, 0.3]
for name, start, n_epochs in [("from0", (0.0, 0.0), 3),
                              ("frombest", tuple(best), 1)]:
    if name == "from0":
        print("\n(1) (0, 0) から 3 エポック。"
              "3 エポック目の損失の平均 - 最小値")
    else:
        print("\n(2) 谷底から 1 エポック。損失の平均 - 最小値（揺れ）")
    print("   B   学習率   中央値   [種 3 つの最小, 最大]   損失の最大")
    for B in [1, 10, 100]:
        rows = []
        for eta in etas:
            res = [run(eta, B, seed, start, n_epochs) for seed in range(3)]
            ex = np.array([ls.mean() - L_min for ls in res])
            top = max(ls.max() for ls in res)
            rows.append((eta, np.median(ex), ex.min(), ex.max(), top))
            print(f"{B:4d}  {eta:<6}  {np.median(ex):.2e}  [{ex.min():.2e}, "
                  f"{ex.max():.2e}]  {top:.4f}")
            if name == "from0" and B == 1 and eta == 0.1:
                fr = [np.mean(ls > 0.04) for ls in res]
                print(f"      損失が 0.04 を超えたステップの割合: "
                      f"{min(fr):.4f}-{max(fr):.4f}")
        np.savetxt(OUT / f"sweep_sgd_{name}_b{B}.csv", rows, delimiter=",",
                   header="eta,excess,lo,hi,max", comments="", fmt="%.6g")
