"""多項式回帰で、正則化の強さとデータの件数を変えたときの変化を調べる。

overfit_poly.py と同じデータ・同じ当てはめ方（fit 関数）を使う。
(1) 9 次式の正則化の強さ lambda を 10 の 0.5 乗刻みで変え、訓練誤差と検証誤差。
(2) lambda = 0, 1e-5, 1e-3, 1 の 4 通りで、当てはめた曲線（図のため）。
(3) 訓練データの件数 n を 6 通りに変え、3 次式と 9 次式の検証誤差。
    データの乱数の種を 10 通り変え、中央値と最小・最大を記録する。
(4) 正則化なしの 9 次式の係数と、各項 w_j x^j の値（振れの理由を見るため）。
検証データは、どの実験でも overfit_poly.py と同じ 200 件。
"""

from pathlib import Path

import numpy as np

from overfit_poly import features, fit, make_data, rmse
from overfit_poly import x_train, x_val, y_train, y_val

OUT = Path(__file__).parent / "out"

# ---- (1) 正則化の強さ lambda を変える（9 次式、訓練データ 10 件） ----
print("(1) 9 次式の lambda と誤差")
print("lambda     訓練誤差  検証誤差  係数の絶対値の最大")
rows = []
for k in np.arange(-8.0, 1.01, 0.5):
    lam = 10.0 ** k
    w = fit(x_train, y_train, 9, lam)
    rows.append((lam, rmse(w, x_train, y_train), rmse(w, x_val, y_val),
                 np.abs(w).max()))
    if k == int(k):  # 表示は 10 倍刻みだけ
        print(f"{lam:<9.0e}  {rows[-1][1]:.3f}    {rows[-1][2]:.3f}"
              f"    {rows[-1][3]:10.1f}")
np.savetxt(OUT / "sweep_poly_lam.csv", rows, delimiter=",",
           header="lam,train,val,maxcoef", comments="", fmt="%.6g")
best = min(rows, key=lambda r: r[2])
print(f"検証誤差が最小の lambda: {best[0]:.0e}（検証誤差 {best[2]:.3f}）")
w0 = fit(x_train, y_train, 9)
val0 = rmse(w0, x_val, y_val)
print(f"lambda = 0（正則化なし）の検証誤差: {val0:.3f}")
# 本当の曲線そのものの検証誤差（偶然のずれだけが残る）
val_true = np.sqrt(np.mean((np.sin(np.pi * x_val) - y_val) ** 2))
print(f"本当の曲線 sin(πx) の検証誤差: {val_true:.3f}")
with open(OUT / "sweep_poly_lam0.csv", "w") as f:  # 図の水平線のため
    f.write("x,val0,true\n")
    for xx in ["1e-9", "1", "1e2", "1e4"]:  # lambda の軸と n の軸の両端
        f.write(f"{xx},{val0:.6g},{val_true:.6g}\n")

# ---- (2) 4 通りの lambda で当てはめた曲線 ----
grid = np.linspace(-1.0, 1.0, 201)
cols = {"x": grid, "true": np.sin(np.pi * grid)}
for name, lam in [("lam0", 0.0), ("lam1e-5", 1e-5), ("lam1e-3", 1e-3),
                  ("lam1", 1.0)]:
    cols[name] = features(grid, 9) @ fit(x_train, y_train, 9, lam)
np.savetxt(OUT / "sweep_poly_curves.csv", np.stack(list(cols.values()), 1),
           delimiter=",", header=",".join(cols), comments="", fmt="%.5f")

# ---- (3) 訓練データの件数 n を変える（種を 10 通り） ----
print("\n(3) 訓練データの件数と検証誤差（種 10 通りの中央値 [最小, 最大]）")
print("     n   3 次式                 9 次式")
ns = [10, 30, 50, 100, 300, 1000]
for M in [3, 9]:
    with open(OUT / f"sweep_poly_n_m{M}.csv", "w") as f:
        f.write("n,median,lo,hi\n")
        for n in ns:
            errs = []
            for seed in range(10):
                xs, ys = make_data(n, np.random.default_rng(100 + seed))
                errs.append(rmse(fit(xs, ys, M), x_val, y_val))
            f.write(f"{n},{np.median(errs):.6g},{min(errs):.6g},"
                    f"{max(errs):.6g}\n")
res = {M: np.loadtxt(OUT / f"sweep_poly_n_m{M}.csv", delimiter=",",
                    skiprows=1) for M in [3, 9]}
for r3, r9 in zip(res[3], res[9]):
    print(f"{int(r3[0]):6d}  {r3[1]:.3f} [{r3[2]:.3f}, {r3[3]:.3f}]  "
          f"{r9[1]:.3f} [{r9[2]:.3f}, {r9[3]:.3f}]")

# ---- (4) 9 次式（正則化なし）の係数と、各項の値 ----
x_on = x_train[1]  # 訓練データの点（左から 2 番目）
x_mid = (x_train[0] + x_train[1]) / 2  # 左端の 2 点の中間
print(f"\n(4) 9 次式の係数と、x = {x_on:.3f}（訓練データの点）と"
      f" x = {x_mid:.3f}（中間）での各項 w_j x^j")
print(" j      w_j     w_j x^j（点）  w_j x^j（中間）")
terms_on, terms_mid = w0 * x_on ** np.arange(10), w0 * x_mid ** np.arange(10)
for j in range(10):
    print(f"{j:2d}  {w0[j]:8.1f}   {terms_on[j]:8.2f}"
          f"      {terms_mid[j]:8.2f}")
print(f"和（予測）:   {terms_on.sum():8.3f}      {terms_mid.sum():8.3f}")
print(f"訓練データの正解 y = {y_train[1]:.3f}、本当の曲線 sin(πx): "
      f"{np.sin(np.pi * x_on):.3f}（点）、{np.sin(np.pi * x_mid):.3f}（中間）")
np.savetxt(OUT / "sweep_poly_terms.csv",
           np.stack([np.arange(10), w0, terms_on, terms_mid], 1),
           delimiter=",", header="j,coef,on,mid", comments="", fmt="%.5f")
