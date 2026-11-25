"""ミニバッチの勾配のばらつきが、バッチサイズ B でどう変わるかを調べる。

sgd_minibatch.py の実験 1 と同じ設定（1 万件のデータ、w = b = 0 の地点、
B 件を無作為に選んで dL/dw を計算することを 2000 回）で、B を 10 通りに変える。
図のために、B = 1, 10, 100 の 2000 回分の勾配のヒストグラムも保存する。
"""

from pathlib import Path

import numpy as np

from linreg_numpy import grad

OUT = Path(__file__).parent / "out"
data = np.loadtxt(OUT / "char_token_big.csv", delimiter=",", skiprows=1)
x, y = data[:, 0] / 100, data[:, 1] / 100
N = len(x)  # 10000
rng = np.random.default_rng(0)

w, b = 0.0, 0.0
full_dw, _ = grad(w, b, x, y)  # 全データ（1 万件）の勾配
print(f"全データの勾配 dL/dw = {full_dw:.4f}")

Bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
samples = {}
rows = []
print("     B    平均     標準偏差   B=1 の標準偏差 / sqrt(B)")
for B in Bs:
    est = np.empty(2000)
    for k in range(2000):  # ミニバッチを 2000 回選び直す
        idx = rng.choice(N, size=B, replace=False)
        est[k], _ = grad(w, b, x[idx], y[idx])
    samples[B] = est
    ref = samples[1].std() / np.sqrt(B)  # 1/sqrt(B) に比例するときの値
    rows.append((B, est.mean(), est.std(), ref))
    print(f"{B:6d}  {est.mean():.4f}  {est.std():.4f}     {ref:.4f}")

np.savetxt(OUT / "sweep_grad_noise.csv", rows, delimiter=",",
           header="B,mean,std,ref", comments="", fmt="%.6g")

# ヒストグラム（縦軸は確率密度。棒の面積の合計が 1）
edges = np.arange(-7.0, 1.0001, 0.1)
centers = (edges[:-1] + edges[1:]) / 2
cols = [centers]
for B in [1, 10, 100]:
    dens, _ = np.histogram(samples[B], bins=edges, density=True)
    cols.append(dens)
    inside = np.mean((samples[B] >= edges[0]) & (samples[B] < edges[-1]))
    print(f"B = {B:3d}: ヒストグラムの範囲に入った割合 {inside:.4f}")
np.savetxt(OUT / "sweep_grad_hist.csv", np.stack(cols, 1), delimiter=",",
           header="center,b1,b10,b100", comments="", fmt="%.5f")
