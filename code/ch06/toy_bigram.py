"""第 6 章: 12 文字の文章で、バイグラム言語モデルを手で作る（手計算の検算）。

実行: cd code && python ch06/toy_bigram.py
"""
import math
from collections import Counter

# [snippet: toy]
text = "すもももももももものうち"
pairs = list(zip(text[:-1], text[1:]))  # (前の文字, 次の文字) の組を作る
print(f"文字数 {len(text)}  組の数 {len(pairs)}  最初の 3 組 {pairs[:3]}")

count = Counter(pairs)                  # 同じ組が何回あったかを数える
total = Counter(a for a, b in pairs)    # 前の文字ごとの合計
p = {(a, b): c / total[a] for (a, b), c in count.items()}
print("  ".join(f"{a}→{b} {q:.3f}" for (a, b), q in p.items()))

# 11 回の予測のそれぞれで -log p（正解に付けた確率の対数）を求め、平均する
loss = -sum(math.log(p[a, b]) for a, b in pairs) / len(pairs)
print(f"損失 {loss:.4f}  パープレキシティ {math.exp(loss):.3f}")
# [end: toy]
