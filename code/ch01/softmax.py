"""softmax と log-sum-exp。最後に、モデルの最後の一歩を再現する。"""

import numpy as np


# [snippet: naive]
def softmax_naive(z):
    e = np.exp(z)  # 各成分を e の肩に乗せる（必ず正になる）
    return e / e.sum()  # 合計で割る（和が 1 になる）


print(softmax_naive(np.array([2.0, 1.0, 0.0])))
print(softmax_naive(np.array([1002.0, 1001.0, 1000.0])))
# [end: naive]

print("---")
# [snippet: stable]
def softmax(z):
    e = np.exp(z - z.max())  # 最大値を引いてから exp。最大の成分は e^0 = 1
    return e / e.sum()

def log_softmax(z):
    m = z.max()
    lse = m + np.log(np.sum(np.exp(z - m)))  # log-sum-exp
    return z - lse  # log p_k = z_k - log Σ exp(z_j)
# [end: stable]

print(softmax(np.array([2.0, 1.0, 0.0])))
print(softmax(np.array([1002.0, 1001.0, 1000.0])))
print(log_softmax(np.array([1002.0, 1001.0, 1000.0])))

print("---")
# [snippet: last-step]
data = np.load("ch01/out/last_step.npz")
h, W, tokens = data["h"], data["W"], data["tokens"].tolist()
z = W @ h  # (1) 内積でスコアを作る: (4096, 192) と (192,) から (4096,)
p = softmax(z)  # (2) スコアを確率に直す
print("確率の和 =", p.sum().round(6))
for i in np.argsort(-p)[:5]:  # (3) 確率の大きい順に 5 個
    print(f"{tokens[i]!r:10} z = {z[i]:5.2f}  p = {p[i]:.3f}")
k = np.argmax(p)
print(f"正解が {tokens[k]!r} のときの損失 = {-log_softmax(z)[k]:.3f}")
print(f"何も知らないモデルの損失 log 4096 = {np.log(4096):.3f}")
# [end: last-step]
