"""二乗誤差と最尤推定の関係を、3 件のデータで数値的に確かめる。"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"

x = np.array([0.4, 0.8, 1.6])  # 文字数（百字単位）
y = np.array([21.0, 42.0, 101.0])  # トークン数（ここではトークン単位）
sigma = 10.0  # ノイズの標準偏差（トークン）


def normal_pdf(y, mu, sigma):
    """正規分布 N(mu, sigma^2) の確率密度。"""
    return np.exp(-(y - mu) ** 2 / (2 * sigma ** 2)) / (
        np.sqrt(2 * np.pi) * sigma)


print("w     各データの確率密度         尤度       負の対数尤度  二乗誤差の和")
# [snippet: likelihood]
for w in [0.5, 0.6, 0.7]:
    mu = 100 * w * x  # モデルの予測（トークン単位）
    dens = normal_pdf(y, mu, sigma)  # 3 件それぞれの確率密度
    likelihood = np.prod(dens)  # 尤度: 3 つの積
    nll = -np.sum(np.log(dens))  # 負の対数尤度
    sse = np.sum((y - mu) ** 2)  # 二乗誤差の和
    print(f"{w:.1f}  {dens[0]:.4f} {dens[1]:.4f} {dens[2]:.4f}  "
          f"{likelihood:.2e}  {nll:10.3f}  {sse:10.1f}")
# [end: likelihood]

# [snippet: argmax]
ws = np.linspace(0.0, 1.2, 1201)  # w の候補を 0.001 刻みで並べる
nlls = [-np.sum(np.log(normal_pdf(y, 100 * w * x, sigma))) for w in ws]
sses = [np.sum((y - 100 * w * x) ** 2) for w in ws]
print(f"尤度が最大になる w     = {ws[np.argmin(nlls)]:.3f}")
print(f"二乗誤差が最小になる w = {ws[np.argmin(sses)]:.3f}")
# [end: argmax]

if __name__ == "__main__":
    # 1 万件のデータで、当てはめた直線からのずれ（残差）の分布を調べる
    data = np.loadtxt(OUT / "char_token_big.csv", delimiter=",", skiprows=1)
    xb, yb = data[:, 0] / 100, data[:, 1]
    A = np.stack([xb, np.ones(len(xb))], axis=1)
    theta = np.linalg.lstsq(A, yb, rcond=None)[0]
    resid = yb - A @ theta  # 正解 - 予測（トークン単位）
    print(f"\n残差の標準偏差: {resid.std():.2f} トークン")
    inside = np.mean(np.abs(resid) <= resid.std())
    print(f"残差が ±1 標準偏差に入る割合: {inside:.3f}（正規分布なら 0.683）")
    edges = np.arange(-42, 43, 4)
    hist, _ = np.histogram(resid, bins=edges, density=True)
    with open(OUT / "resid_hist.csv", "w") as f:
        f.write("center,density\n")
        for c, h in zip((edges[:-1] + edges[1:]) / 2, hist):
            f.write(f"{c:.1f},{h:.6f}\n")
