"""ミニバッチ確率的勾配降下法（SGD）を、全データを使う勾配降下法と比べる。

題材: 文字数 → トークン数のデータ 1 万件（make_data.py が作る）。
"""

from pathlib import Path

import numpy as np

from linreg_numpy import grad, mse, save_contours

OUT = Path(__file__).parent / "out"
data = np.loadtxt(OUT / "char_token_big.csv", delimiter=",", skiprows=1)
x, y = data[:, 0] / 100, data[:, 1] / 100
N = len(x)  # 10000
rng = np.random.default_rng(0)

# ---- 実験 1: ミニバッチの勾配は、全データの勾配の「見積もり」である ----
# [snippet: grad-noise]
w, b = 0.0, 0.0
full_dw, _ = grad(w, b, x, y)  # 全データ（1 万件）を使った勾配
print(f"全データの勾配 dL/dw = {full_dw:.4f}")
print("     B    平均     標準偏差")
for B in [1, 10, 100, 1000]:
    estimates = []
    for _ in range(2000):  # ミニバッチを 2000 回選び直す
        idx = rng.choice(N, size=B, replace=False)  # B 件をランダムに選ぶ
        dw, _ = grad(w, b, x[idx], y[idx])  # 選んだ B 件だけで勾配を計算
        estimates.append(dw)
    print(f"{B:6d}  {np.mean(estimates):.4f}  {np.std(estimates):.4f}")
# [end: grad-noise]


# ---- 実験 2: 同じ「データ処理数」でどこまで損失が下がるか ----
# [snippet: sgd]
def sgd(x, y, eta, B, n_epochs, w=0.0, b=0.0):
    """ミニバッチ SGD。(処理したデータ数, w, b) の記録を返す。"""
    N = len(x)
    history = [(0, w, b)]
    seen = 0  # これまでに処理したデータ数
    # [snippet: sgd-loop]
    for epoch in range(n_epochs):
        perm = rng.permutation(N)  # エポックごとにデータの順番を混ぜる
        for start in range(0, N, B):
            idx = perm[start:start + B]  # 先頭から B 件ずつ取り出す
            dw, db = grad(w, b, x[idx], y[idx])  # ミニバッチの勾配
            w = w - eta * dw
            b = b - eta * db
            # [end: sgd-loop]
            seen += len(idx)
            history.append((seen, w, b))
    return history
# [end: sgd]


if __name__ == "__main__":
    eta = 0.1
    runs = {
        "full": sgd(x, y, eta, B=N, n_epochs=300),  # B = N は勾配降下法
        "b100": sgd(x, y, eta, B=100, n_epochs=3),
        "b10": sgd(x, y, eta, B=10, n_epochs=3),
        "b1": sgd(x, y, eta, B=1, n_epochs=3),
    }
    best = np.linalg.lstsq(np.stack([x, np.ones(N)], 1), y, rcond=None)[0]
    print(f"\n到達できる最小の損失: {mse(best[0], best[1], x, y):.5f}")
    print("データ処理数と損失（損失は全 1 万件で評価）")
    print("  処理数     全データ   B=100     B=10      B=1")
    for seen in [1000, 10000, 30000, 100000, 1000000, 3000000]:
        cells = []
        for name, B in [("full", N), ("b100", 100), ("b10", 10), ("b1", 1)]:
            k = seen // B
            h = runs[name]
            if seen % B == 0 and 1 <= k < len(h):
                cells.append(f"{mse(h[k][1], h[k][2], x, y):.5f}")
            else:
                cells.append("   -   ")
        print(f"{seen:8d}   " + "   ".join(cells))

    # ---- 図のためのデータ出力 ----
    for name, B in [("full", N), ("b100", 100), ("b10", 10), ("b1", 1)]:
        h = runs[name]
        with open(OUT / f"sgd_curve_{name}.csv", "w") as f:
            f.write("seen,loss\n")
            # 対数軸で見やすいよう、記録を間引く
            ks = sorted(set(np.unique(np.geomspace(1, len(h) - 1, 120)
                                      .astype(int)).tolist()))
            for k in ks:
                f.write(f"{h[k][0]},{mse(h[k][1], h[k][2], x, y):.6f}\n")
    for name, n in [("full", 300), ("b10", 300), ("b1", 300)]:
        with open(OUT / f"sgd_path_{name}.csv", "w") as f:
            f.write("step,w,b\n")
            for k, (_, w_k, b_k) in enumerate(runs[name][:n + 1]):
                f.write(f"{k},{w_k:.5f},{b_k:.5f}\n")
    save_contours(OUT / "loss_contours_big.csv", x, y,
                  [0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
