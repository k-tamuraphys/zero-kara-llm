"""サンプリング・期待値・分散・正規分布を数値実験で確かめる。"""

import numpy as np

rng = np.random.default_rng(0)  # 乱数の種を固定する

# [snippet: sample]
tokens = np.load("ch01/out/last_step.npz")["tokens"].tolist()
p = np.load("ch01/out/contexts.npz")["shikata"]  # 「どうも仕方が」の次の分布
samples = rng.choice(len(p), size=10000, p=p)  # 分布 p に従って 1 万回引く
print("最初の 8 回:", [tokens[i] for i in samples[:8]])
k = np.argmax(p)  # 確率が最大のトークン（「ない。」）
for n in [10, 100, 1000, 10000]:
    freq = np.mean(samples[:n] == k)  # 最初の n 回のうち「ない。」の割合
    print(f"n = {n:5d}: 「ない。」の割合 = {freq:.3f}")
print(f"確率 p = {p[k]:.3f}")
# [end: sample]

# 図用: 引いた回数 n と、「ない。」が出た割合
hits = np.cumsum(samples == k)
with open("ch01/out/sampling.csv", "w") as fp:
    fp.write("n,freq\n")
    for n in np.unique(np.logspace(0, 4, 200).astype(int)):
        fp.write(f"{n},{hits[n - 1] / n:.4f}\n")

print("---")
# [snippet: moments]
x = np.arange(1, 7)  # サイコロの目 1, 2, ..., 6
p = np.ones(6) / 6  # どの目も確率 1/6
mean = np.sum(p * x)  # 期待値: 値 × 確率 の合計
var = np.sum(p * (x - mean) ** 2)  # 分散: (値 - 期待値)^2 の期待値
print(f"期待値 = {mean:.3f}, 分散 = {var:.3f}, 標準偏差 = {np.sqrt(var):.3f}")
rolls = rng.choice(x, size=100000, p=p)  # 実際に 10 万回振る
print(f"10 万回の平均 = {rolls.mean():.3f}, 分散 = {rolls.var():.3f}")

s = rng.normal(loc=0.0, scale=0.02, size=100000)  # 平均 0、標準偏差 0.02
print(f"平均 = {s.mean():+.5f}, 標準偏差 = {s.std():.5f}")
for c in [1, 2, 3]:
    inside = np.mean(np.abs(s) < c * 0.02)  # 平均から標準偏差の c 倍以内の割合
    print(f"±{c}σ 以内の割合 = {inside:.3f}")
# [end: moments]

# 図用: 標準偏差を単位にしたヒストグラム（面積が 1 になるように正規化）
hist, edges = np.histogram(s / 0.02, bins=40, range=(-4, 4), density=True)
with open("ch01/out/normal_hist.csv", "w") as fp:
    fp.write("x,density\n")
    for c, v in zip((edges[:-1] + edges[1:]) / 2, hist):
        fp.write(f"{c:.2f},{v:.4f}\n")
