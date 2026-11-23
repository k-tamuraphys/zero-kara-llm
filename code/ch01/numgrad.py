"""数値微分で、勾配と連鎖律を確かめる。"""

import numpy as np


# [snippet: numgrad]
def f(x, y):
    return x**2 + 2 * y**2

def numerical_grad(f, x, y, eps=1e-5):
    """(x, y) での勾配を、少し動かしたときの変化から求める。"""
    dfdx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)  # x だけ動かす
    dfdy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)  # y だけ動かす
    return np.array([dfdx, dfdy])

g = numerical_grad(f, 1.0, 1.0)
print("数値微分:", g.round(6))
print("手計算  :", np.array([2 * 1.0, 4 * 1.0]))  # (2x, 4y)
# [end: numgrad]

print("---")
# [snippet: direction]
step = 0.01  # 動かす距離
for deg in range(0, 360, 45):  # 向きを 45 度ずつ変える
    theta = np.deg2rad(deg)
    u = np.array([np.cos(theta), np.sin(theta)])  # その向きの長さ 1 のベクトル
    df = f(1.0 + step * u[0], 1.0 + step * u[1]) - f(1.0, 1.0)
    dot = g @ u * step  # 勾配と「動かした分」の内積
    print(f"{deg:3d} 度: f の増加 {df:+.4f}   勾配との内積 {dot:+.4f}")
# [end: direction]

# 図用に 1 度刻みで調べ、最も増える向きを探す
degs = np.arange(0, 361)
u = np.stack([np.cos(np.deg2rad(degs)), np.sin(np.deg2rad(degs))], axis=1)
dfs = f(1.0 + step * u[:, 0], 1.0 + step * u[:, 1]) - f(1.0, 1.0)
with open("ch01/out/direction.csv", "w") as fp:
    fp.write("deg,df\n")
    for d, v in zip(degs, dfs):
        fp.write(f"{d},{v:.6f}\n")
print("最も増えた向き:", degs[np.argmax(dfs)], "度")
print("勾配の向き    :", np.rad2deg(np.arctan2(g[1], g[0])).round(1), "度")

print("---")
# [snippet: chain]
x, y = 2.0, 7.0  # データ（固定）
w, b = 1.5, 0.5  # パラメータ


def loss(w, b):
    return (w * x + b - y) ** 2


y_hat = w * x + b  # 予測
r = y_hat - y  # 予測と正解のずれ
dL_dr = 2 * r  # L = r^2 を r で微分
dL_dw = dL_dr * 1.0 * x  # 連鎖律: dL/dr * dr/dy_hat * dy_hat/dw
dL_db = dL_dr * 1.0 * 1.0  # 連鎖律: dL/dr * dr/dy_hat * dy_hat/db
eps = 1e-5
num_w = (loss(w + eps, b) - loss(w - eps, b)) / (2 * eps)
num_b = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)
print(f"L = {loss(w, b)}")
print(f"dL/dw: 連鎖律 {dL_dw:.4f}  数値微分 {num_w:.4f}")
print(f"dL/db: 連鎖律 {dL_db:.4f}  数値微分 {num_b:.4f}")
# [end: chain]

# 図用: 刻み幅 eps と数値微分の誤差（exp(x) の x = 1 での微分。真の値は e）
with open("ch01/out/eps_error.csv", "w") as fp:
    fp.write("eps,forward,central\n")
    for k in np.arange(1, 13.01, 0.5):
        eps = 10.0 ** (-k)
        fwd = (np.exp(1 + eps) - np.exp(1)) / eps
        cen = (np.exp(1 + eps) - np.exp(1 - eps)) / (2 * eps)
        fp.write(f"{eps:.3e},{abs(fwd - np.e):.3e},{abs(cen - np.e):.3e}\n")
