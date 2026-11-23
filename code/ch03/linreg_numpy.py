"""線形回帰を勾配降下法で解く（NumPy、勾配は手で導いた式）。

題材: 漱石コーパスの断片の「文字数 → トークン数」（make_data.py が作る）。
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"

# [snippet: load]
data = np.loadtxt(OUT / "char_token_train.csv", delimiter=",", skiprows=1)
x = data[:, 0] / 100  # 文字数（百字単位）      形 (N,)
y = data[:, 1] / 100  # トークン数（百トークン単位） 形 (N,)
N = len(x)  # データ数 100
# [end: load]


# [snippet: model-loss]
def predict(w, b, x):
    """モデル f(x) = w x + b。x が配列なら、全データの予測を一度に返す。"""
    return w * x + b


def mse(w, b, x, y):
    """平均二乗誤差（損失関数）。"""
    return np.mean((predict(w, b, x) - y) ** 2)


def grad(w, b, x, y):
    """損失の勾配 (dL/dw, dL/db) を、手で導いた式で計算する。"""
    err = predict(w, b, x) - y  # 予測 - 正解      形 (N,)
    dw = 2 * np.mean(err * x)  # 「誤差 × 入力」の平均の 2 倍
    db = 2 * np.mean(err)  # 「誤差」の平均の 2 倍
    return dw, db
# [end: model-loss]


# [snippet: numerical-check]
def numerical_grad(w, b, x, y, h=1e-6):
    """数値微分: パラメータを少しだけ動かして損失の変化率を測る。"""
    dw = (mse(w + h, b, x, y) - mse(w - h, b, x, y)) / (2 * h)
    db = (mse(w, b + h, x, y) - mse(w, b - h, x, y)) / (2 * h)
    return dw, db
# [end: numerical-check]


# [snippet: gd]
def gradient_descent(x, y, eta, n_steps, w=0.0, b=0.0):
    """勾配降下法。各ステップの (w, b, 損失) を記録して返す。"""
    history = [(w, b, mse(w, b, x, y))]
    for step in range(n_steps):
        dw, db = grad(w, b, x, y)  # (1) いまの場所での勾配
        w = w - eta * dw  # (2) 勾配の逆向きに少し動く
        b = b - eta * db
        history.append((w, b, mse(w, b, x, y)))
    return history
# [end: gd]


def save_contours(path, x, y, levels, n=121):
    """図のために、損失の等高線（楕円）の座標を保存する。

    損失は (w, b) の 2 次関数なので、等高線は最小点を中心とする楕円になる。
    """
    A = np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
    X = np.stack([x, np.ones(len(x))], axis=1)
    best = np.linalg.solve(X.T @ X, X.T @ y)  # 最小点
    lmin = mse(best[0], best[1], x, y)  # 最小値
    lam, U = np.linalg.eigh(A)  # 楕円の軸の向きと長さ
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])  # 単位円 (2, n)
    with open(path, "w") as f:
        f.write("w,b,level\n")
        for level in levels:
            radius = np.sqrt((level - lmin) / lam)  # 2 本の軸の半径
            pts = best[:, None] + U @ (circle * radius[:, None])
            for ww, bb in pts.T:
                f.write(f"{ww:.5f},{bb:.5f},{level}\n")
            f.write("\n")


if __name__ == "__main__":
    print("勾配の照合 (w, b) = (0.3, 0.2)")
    print("  手で導いた式:", "%.6f %.6f" % grad(0.3, 0.2, x, y))
    print("  数値微分    :", "%.6f %.6f" % numerical_grad(0.3, 0.2, x, y))

    # [snippet: run]
    history = gradient_descent(x, y, eta=0.1, n_steps=1000)
    for step in [0, 1, 2, 5, 10, 100, 300, 1000]:
        w, b, loss = history[step]
        print(f"step {step:4d}  w = {w:.4f}  b = {b:+.4f}  損失 = {loss:.5f}")
    # [end: run]

    # 解析解（正規方程式）と比べる
    # [snippet: normal-eq]
    X = np.stack([x, np.ones(N)], axis=1)  # 各行が (x_i, 1)。形 (N, 2)
    theta = np.linalg.solve(X.T @ X, X.T @ y)  # 正規方程式を解く
    print(f"解析解       w = {theta[0]:.4f}  b = {theta[1]:+.4f}  "
          f"損失 = {mse(theta[0], theta[1], x, y):.5f}")
    # [end: normal-eq]
    w, b, loss = history[-1]
    print(f"100 字の断片の予測: {100 * predict(w, b, 1.0):.1f} トークン")
    print(f"誤差の目安（損失の平方根）: {100 * np.sqrt(loss):.1f} トークン")
    test = np.loadtxt(OUT / "char_token_test.csv", delimiter=",", skiprows=1)
    x_test, y_test = test[:, 0] / 100, test[:, 1] / 100
    print(f"訓練データ 100 件での損失      : {mse(w, b, x, y):.5f}")
    print(f"学習に使っていない 100 件での損失: "
          f"{mse(w, b, x_test, y_test):.5f}")

    # ---- 以下は図のためのデータ出力 ----
    with open(OUT / "linreg_path.csv", "w") as f:  # 勾配降下法の軌跡
        f.write("step,w,b,loss\n")
        for step, (w, b, loss) in enumerate(history[:301]):
            f.write(f"{step},{w:.5f},{b:.5f},{loss:.6f}\n")
    etas = [0.01, 0.1, 0.3, 0.42]  # 学習率の比較
    runs = [gradient_descent(x, y, eta, 100) for eta in etas]
    with open(OUT / "linreg_lr.csv", "w") as f:
        f.write("step," + ",".join(f"eta{e}" for e in etas) + "\n")
        for step in range(101):
            vals = [min(run[step][2], 1e6) for run in runs]
            f.write(f"{step}," + ",".join(f"{v:.6g}" for v in vals) + "\n")
    print("学習率の比較（100 ステップ後の損失）")
    for eta, run in zip(etas, runs):
        print(f"  eta = {eta:<5} 損失 = {run[100][2]:.5g}")
    save_contours(OUT / "loss_contours.csv", x, y,
                  [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
