"""ロジスティック回帰: 断片が 60 トークンに収まるかを文字数から予測する。"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
LIMIT = 60  # トークン数の上限


def load(name):
    data = np.loadtxt(OUT / name, delimiter=",", skiprows=1)
    x = data[:, 0] / 100  # 文字数（百字単位）
    y = (data[:, 1] <= LIMIT).astype(float)  # 収まれば 1、収まらなければ 0
    return x, y


x, y = load("char_token_train.csv")


# [snippet: logistic]
def sigmoid(z):
    """シグモイド関数: どんな実数も 0 と 1 の間に押し込める。"""
    return 1 / (1 + np.exp(-z))


def predict_proba(w, b, x):
    """モデル: 「収まる」確率 p = sigmoid(w x + b)。"""
    return sigmoid(w * x + b)


def bce(w, b, x, y):
    """損失: 負の対数尤度の平均（2 値の交差エントロピー）。"""
    p = predict_proba(w, b, x)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def grad(w, b, x, y):
    """勾配。線形回帰と同じ「誤差 × 入力」の形になる。"""
    err = predict_proba(w, b, x) - y  # 予測確率 - 正解
    return np.mean(err * x), np.mean(err)
# [end: logistic]


if __name__ == "__main__":
    print(f"訓練データ {len(y)} 件のうち、収まる（y = 1）のは "
          f"{int(y.sum())} 件")
    # [snippet: train]
    w, b, eta = 0.0, 0.0, 1.0
    for step in range(5001):
        if step in (0, 10, 100, 1000, 5000):
            acc = np.mean((predict_proba(w, b, x) > 0.5) == (y == 1))
            print(f"step {step:4d}  w = {w:7.3f}  b = {b:7.3f}  "
                  f"損失 = {bce(w, b, x, y):.4f}  正解率 = {acc:.2f}")
        dw, db = grad(w, b, x, y)
        w, b = w - eta * dw, b - eta * db
    # [end: train]

    print(f"確率がちょうど 0.5 になる文字数: {-100 * b / w:.1f} 字")
    for chars in [60, 90, 100, 110, 140]:
        p = predict_proba(w, b, chars / 100)
        print(f"  {chars:3d} 字の断片が収まる確率: {p:.3f}")
    x_test, y_test = load("char_token_test.csv")
    acc = np.mean((predict_proba(w, b, x_test) > 0.5) == (y_test == 1))
    print(f"学習に使っていない 100 件での正解率: {acc:.2f}")

    with open(OUT / "logistic_points.csv", "w") as f:
        f.write("chars,y\n")
        for xi, yi in zip(x, y):
            f.write(f"{100 * xi:.0f},{int(yi)}\n")
    with open(OUT / "logistic_curve.csv", "w") as f:
        f.write("chars,p\n")
        for c in np.linspace(20, 200, 181):
            f.write(f"{c:.0f},{predict_proba(w, b, c / 100):.5f}\n")
