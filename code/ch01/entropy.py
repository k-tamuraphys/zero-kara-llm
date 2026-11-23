"""エントロピー・交差エントロピー・KL ダイバージェンスを数値で確かめる。"""

import numpy as np


# [snippet: entropy]
def entropy(p):
    p = p[p > 0]  # 確率 0 の項は除く（0 log 0 = 0 と約束する）
    return np.sum(p * np.log(1 / p))  # 驚き log(1/p) の期待値

def cross_entropy(p, q):
    return -np.sum(p * np.log(q))  # 現実は p、予測は q

def kl(p, q):
    return np.sum(p * np.log(p / q))

q = np.ones(6) / 6  # 公平なサイコロ
p = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])  # 1 の目が出やすいサイコロ
bad = np.array([0.001, 0.1998, 0.1998, 0.1998, 0.1998, 0.1998])
print(f"H(q) = {entropy(q):.3f}  H(p) = {entropy(p):.3f}")
print(f"H(p, q) = {cross_entropy(p, q):.3f}  KL(p||q) = {kl(p, q):.3f}")
print(f"KL(q||p) = {kl(q, p):.3f}  KL(p||bad) = {kl(p, bad):.3f}")
# [end: entropy]

print("---")
# [snippet: real]
P = np.load("ch01/out/contexts.npz")  # 実物のモデルが出した確率分布
tokens = np.load("ch01/out/last_step.npz")["tokens"].tolist()
for name, text in [("drill47", "12+35=47"), ("shikata", "どうも仕方が"),
                   ("wagahai", "吾輩は")]:
    p = P[name]
    k = np.argmax(p)
    print(f"H = {entropy(p):.3f}  最大の確率 {p[k]:.3f} {tokens[k]!r:8}"
          f"  文脈: {text}")
print(f"上限 log 4096 = {np.log(4096):.3f}")
p = P["shikata"]
k = tokens.index("ない。")  # 「ない。」の番号
print(f"正解が「ない。」のときの損失 -log p = {-np.log(p[k]):.3f}")
# [end: real]
