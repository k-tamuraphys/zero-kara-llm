"""計算ドリル 1000 問を数えて、同時分布・周辺分布・条件付き分布を確かめる。"""

import random

import numpy as np

from llmbook.arith import make_problem
from llmbook.corpus import load_corpus

# [snippet: joint]
rng = random.Random(0)
counts = np.zeros((2, 4))  # 行: 演算子（+, -）、列: 答えの桁数（1〜4）
for _ in range(1000):
    question, answer = make_problem(rng)  # 例: ("12+35=", "47")
    i = 0 if "+" in question else 1
    j = len(answer) - 1
    counts[i, j] += 1
P = counts / counts.sum()  # 同時分布 p(x, y)。全部足すと 1
p_x = P.sum(axis=1)  # 周辺分布 p(x): 横に足す
p_y = P.sum(axis=0)  # 周辺分布 p(y): 縦に足す
print("p(x, y) =")
print(P)
print("p(x) =", p_x)
print("p(y) =", p_y)
# [end: joint]

print("---")
# [snippet: conditional]
p_y_given_minus = P[1] / p_x[1]  # p(y | x = 引き算): 下の行を、行の和で割る
print("p(y | -)    =", p_y_given_minus.round(3))
p_x_given_1 = P[:, 0] / p_y[0]  # p(x | y = 1 桁): 左の列を、列の和で割る
print("p(x | 1 桁) =", p_x_given_1.round(3))
bayes = p_y_given_minus[0] * p_x[1] / p_y[0]  # ベイズの定理
print("ベイズの定理による p(- | 1 桁) =", bayes.round(3))
print("独立なら p(x)p(y) =")
print(np.outer(p_x, p_y).round(3))  # p(x) と p(y) の積を並べた表
# [end: conditional]

print("---")
# [snippet: chain-rule]
text = load_corpus()  # 漱石の小説 14 作品（約 240 万字）
n = len(text)
c1, c2, c3 = text.count("吾"), text.count("吾輩"), text.count("吾輩は")
p1 = c1 / n  # p(吾)
p2 = c2 / c1  # p(輩 | 吾)
p3 = c3 / c2  # p(は | 吾輩)
print(f"回数: 吾 {c1}, 吾輩 {c2}, 吾輩は {c3}（全 {n} 字）")
print(f"p(吾) = {p1:.6f}, p(輩|吾) = {p2:.3f}, p(は|吾輩) = {p3:.3f}")
print(f"3 つの積 = {p1 * p2 * p3:.3e}   直接数えた値 = {c3 / n:.3e}")
# [end: chain-rule]
