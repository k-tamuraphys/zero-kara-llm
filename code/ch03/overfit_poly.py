"""多項式回帰で、未学習・過学習・正則化を観察する。

データは人工的に作る: y = sin(πx) + ノイズ。正解の曲線を知っているので、
モデルがどれだけ「本当の関係」を捉えたかを確かめられる。
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"


# [snippet: data]
def make_data(n, rng, even=True):
    """y = sin(πx) + ノイズ のデータを n 件作る。x の範囲は [-1, 1]。"""
    if even:
        x = np.linspace(-1.0, 1.0, n)  # 等間隔に並べる
    else:
        x = rng.uniform(-1.0, 1.0, size=n)  # ランダムに選ぶ
    y = np.sin(np.pi * x) + rng.normal(scale=0.3, size=n)  # ノイズを足す
    return x, y


rng = np.random.default_rng(1)
x_train, y_train = make_data(10, rng)  # 訓練データはわずか 10 件
x_val, y_val = make_data(200, rng, even=False)  # 検証データ 200 件
# [end: data]


# [snippet: fit]
def features(x, M):
    """x から (1, x, x^2, ..., x^M) を並べた行列を作る。形 (N, M+1)。"""
    return np.stack([x ** j for j in range(M + 1)], axis=1)


def fit(x, y, M, lam=0.0):
    """M 次多項式の係数を最小二乗法で求める（lam > 0 なら L2 正則化つき）。"""
    Phi = features(x, M)
    reg = lam * len(x) * np.eye(M + 1)
    reg[0, 0] = 0.0  # 定数項 w_0 は正則化しない
    return np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)


def rmse(w, x, y):
    """誤差の大きさ: 平均二乗誤差の平方根。"""
    return np.sqrt(np.mean((features(x, len(w) - 1) @ w - y) ** 2))
# [end: fit]


if __name__ == "__main__":
    # [snippet: degree-sweep]
    print("次数 M  訓練誤差  検証誤差  係数の絶対値の最大")
    for M in range(10):
        w = fit(x_train, y_train, M)
        print(f"{M:5d}   {rmse(w, x_train, y_train):.3f}    "
              f"{rmse(w, x_val, y_val):.3f}    {np.abs(w).max():12.1f}")
    # [end: degree-sweep]

    # [snippet: reg-sweep]
    print("\n9 次多項式に L2 正則化をかける")
    print("lambda    訓練誤差  検証誤差  係数の絶対値の最大")
    for lam in [0.0, 1e-5, 1e-3, 1e-1, 1.0]:
        w = fit(x_train, y_train, 9, lam)
        print(f"{lam:<8g}  {rmse(w, x_train, y_train):.3f}    "
              f"{rmse(w, x_val, y_val):.3f}    {np.abs(w).max():12.1f}")
    # [end: reg-sweep]

    print("\n9 次多項式のまま、訓練データを増やす")
    print("訓練データ数  訓練誤差  検証誤差")
    more = {10: (x_train, y_train)}
    for n in [30, 100, 1000]:
        more[n] = make_data(n, rng)
    for n, (xs, ys) in more.items():
        w = fit(xs, ys, 9)
        print(f"{n:10d}    {rmse(w, xs, ys):.3f}    "
              f"{rmse(w, x_val, y_val):.3f}")

    # 検証誤差で次数を選び、選ぶことに使っていないテストデータで測る
    # （既存の乱数列を変えないよう、テストデータは別の種で作る）
    x_test, y_test = make_data(200, np.random.default_rng(2), even=False)
    val_err = [rmse(fit(x_train, y_train, M), x_val, y_val)
               for M in range(10)]
    M_best = int(np.argmin(val_err))
    w = fit(x_train, y_train, M_best)
    print(f"\n検証誤差が最小の次数: M = {M_best}"
          f"（検証誤差 {val_err[M_best]:.3f}）")
    print(f"その多項式のテスト誤差（別の 200 件）: "
          f"{rmse(w, x_test, y_test):.3f}")

    # ---- 図のためのデータ出力 ----
    grid = np.linspace(-1.0, 1.0, 201)
    cols = {"x": grid, "true": np.sin(np.pi * grid)}
    for M in [1, 3, 9]:
        cols[f"m{M}"] = features(grid, M) @ fit(x_train, y_train, M)
    for name, lam in [("reg_good", 1e-3), ("reg_big", 1.0)]:
        cols[name] = features(grid, 9) @ fit(x_train, y_train, 9, lam)
    xs, ys = more[100]
    cols["m9_n100"] = features(grid, 9) @ fit(xs, ys, 9)
    np.savetxt(OUT / "poly_curves.csv", np.stack(list(cols.values()), 1),
               delimiter=",", header=",".join(cols), comments="", fmt="%.5f")
    np.savetxt(OUT / "poly_train.csv", np.stack([x_train, y_train], 1),
               delimiter=",", header="x,y", comments="", fmt="%.5f")
    np.savetxt(OUT / "poly_train100.csv", np.stack([xs, ys], 1),
               delimiter=",", header="x,y", comments="", fmt="%.5f")
    rows = [(M, rmse(fit(x_train, y_train, M), x_train, y_train),
             rmse(fit(x_train, y_train, M), x_val, y_val)) for M in range(10)]
    np.savetxt(OUT / "poly_errors.csv", rows, delimiter=",",
               header="M,train,val", comments="", fmt="%.5f")
