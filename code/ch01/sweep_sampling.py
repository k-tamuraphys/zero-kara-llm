"""第 1 章（図の追加）: 乱数の種を変えて、サンプリングの割合のぶれを見る。

sampling.py と同じく、「どうも仕方が」の次の分布から 1 万回サンプリングし、
最初の n 回のうち「ない。」が出た割合を記録する。これを種 20 通りで行う。
分布と回数は固定し、変えるのは乱数の種だけ。
    cd code && python ch01/sweep_sampling.py > ch01/out/sweep_sampling.txt
出力: ch01/out/sweep_sampling.csv（列 n, s0, ..., s19, lo, hi）
"""
import numpy as np

p = np.load("ch01/out/contexts.npz")["shikata"]  # 「どうも仕方が」の次の分布
k = np.argmax(p)  # 確率が最大のトークン（「ない。」）
SEEDS, N = 20, 10000  # 種の数、1 本あたりのサンプリングの回数
ns = np.logspace(0, 4, 120).astype(int)  # 記録する n（対数で等間隔）
ns = np.unique(np.concatenate([ns, [10, 100, 1000, 10000]]))

curves = []
for seed in range(SEEDS):
    rng = np.random.default_rng(seed)  # 種だけを変える
    samples = rng.choice(len(p), size=N, p=p)
    hits = np.cumsum(samples == k)  # 最初の n 回に「ない。」が出た回数
    curves.append(hits[ns - 1] / ns)  # 割合
curves = np.array(curves)  # (SEEDS, len(ns))
lo, hi = curves.min(axis=0), curves.max(axis=0)

print(f"確率 p = {p[k]:.3f}、種 {SEEDS} 通り")
for n in [10, 100, 1000, 10000]:
    i = np.searchsorted(ns, n)  # ns の中の n の位置
    print(f"n = {n:5d}: 割合の最小 {lo[i]:.3f}  最大 {hi[i]:.3f}"
          f"  幅 {hi[i] - lo[i]:.3f}")

with open("ch01/out/sweep_sampling.csv", "w") as fp:
    cols = ",".join(f"s{s}" for s in range(SEEDS))
    fp.write(f"n,{cols},lo,hi\n")
    for j, n in enumerate(ns):
        vals = ",".join(f"{v:.4f}" for v in curves[:, j])
        fp.write(f"{n},{vals},{lo[j]:.4f},{hi[j]:.4f}\n")
