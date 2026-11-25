"""学習率 eta を細かく変えて、勾配降下法の結果がどう変わるかを調べる。

(1) 3 件のデータ・モデル f(x) = w x（3.3 節の例）で、5 / 50 ステップ後の損失。
(2) 100 件のデータ・モデル f(x) = w x + b（linreg_numpy.py と同じ）で、
    100 ステップ後の損失。
どちらも出発点は 0。変えるのは学習率だけで、ほかは固定する。
発散の境目も、損失の 2 次の係数から計算して表示する。
"""

from pathlib import Path

import numpy as np

# x, y は 100 件のデータ（百字・百トークン単位）
from linreg_numpy import grad, mse, x, y

OUT = Path(__file__).parent / "out"

# ---- (1) 3 件のデータ、パラメータは w だけ ----
x3 = np.array([0.4, 0.8, 1.6])  # 40 字, 80 字, 160 字
y3 = np.array([0.21, 0.42, 1.01])


def gd_1d(eta, n_steps):
    """f(x) = w x の勾配降下法。n_steps 後の損失を返す。"""
    w = 0.0
    for _ in range(n_steps):
        w = w - eta * np.mean(2 * (w * x3 - y3) * x3)
    return np.mean((w * x3 - y3) ** 2)


# 損失は a w^2 + ... なので、谷底からの距離は 1 ステップで (1 - 2 a eta) 倍。
# この倍率の絶対値が 1 を超えると発散する。境目は eta = 1 / a。
a = np.mean(x3 * x3)
print(f"(1) 3 件: w^2 の係数 a = {a:.4f}、発散の境目 eta = 1/a = {1 / a:.3f}")
etas1 = np.round(np.arange(0.02, 1.1001, 0.02), 2)
with open(OUT / "sweep_lr_1d.csv", "w") as f:
    f.write("eta,loss5,loss50\n")
    for eta in etas1:
        f.write(f"{eta:.2f},{gd_1d(eta, 5):.6g},{gd_1d(eta, 50):.6g}\n")
for eta in [0.05, 0.4, 0.8, 0.88, 0.9, 1.0]:
    print(f"    eta = {eta:<5}  損失  5 ステップ後 {gd_1d(eta, 5):.4f}"
          f"  50 ステップ後 {gd_1d(eta, 50):.4g}")


# ---- (2) 100 件のデータ、パラメータは w と b ----
def gd_2d(eta, n_steps):
    """f(x) = w x + b の勾配降下法。n_steps 後の損失を返す。"""
    w, b = 0.0, 0.0
    for _ in range(n_steps):
        dw, db = grad(w, b, x, y)
        w, b = w - eta * dw, b - eta * db
    return mse(w, b, x, y)


# 損失の 2 階微分を並べた行列（ヘッセ行列）の最大の固有値が、
# いちばん急な向きの「谷の急さ」。境目は eta = 2 / 最大の固有値。
H = 2 * np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
lam_max = np.linalg.eigvalsh(H).max()
print(f"(2) 100 件: 最も急な向きの 2 階微分 = {lam_max:.4f}、"
      f"発散の境目 eta = {2 / lam_max:.3f}")
etas2 = np.round(np.arange(0.01, 0.4501, 0.01), 2)
with open(OUT / "sweep_lr_2d.csv", "w") as f:
    f.write("eta,loss100\n")
    for eta in etas2:
        f.write(f"{eta:.2f},{gd_2d(eta, 100):.6g}\n")
for eta in [0.01, 0.1, 0.3, 0.39, 0.40, 0.41, 0.42]:
    print(f"    eta = {eta:<5}  100 ステップ後の損失 = {gd_2d(eta, 100):.4g}")
