"""第 1 章（図の追加）: 表 1.4 の三つの文脈の、次のトークンの分布の上位 10 個。

dump_base.py が保存した確率分布（contexts.npz）を読むだけ。
モデルは動かさない。
    cd code && python ch01/fig_entropy_ctx.py > ch01/out/fig_entropy_ctx.txt
出力: ch01/out/fig_entropy_ctx.csv（列 rank と、文脈ごとの確率）
"""
import numpy as np

P = np.load("ch01/out/contexts.npz")
tokens = np.load("ch01/out/last_step.npz")["tokens"].tolist()
NAMES = ["drill47", "shikata", "wagahai"]  # 12+35=47、どうも仕方が、吾輩は
TOP = 10

tops = []
for name in NAMES:
    p = P[name]
    order = np.argsort(-p)[:TOP]  # 確率の大きい順に 10 個の番号
    q = p[p > 0]
    H = np.sum(q * np.log(1 / q))  # エントロピー（entropy.py と同じ式）
    tops.append(p[order])
    first3 = " ".join(repr(tokens[i]) for i in order[:3])
    print(f"{name:8s} H = {H:.3f}  上位 10 個の合計 = {p[order].sum():.3f}"
          f"  上位 3 個: {first3}")

with open("ch01/out/fig_entropy_ctx.csv", "w") as fp:
    fp.write("rank," + ",".join(NAMES) + "\n")
    for r in range(TOP):
        fp.write(f"{r + 1}," + ",".join(f"{t[r]:.4f}" for t in tops) + "\n")
