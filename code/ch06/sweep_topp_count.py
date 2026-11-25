"""第 6 章（図の追加）: top-p が残す候補の数は、文脈によってどれだけ
変わるのかを数える。top-k は、どの文脈でも同じ k 個を残す。

モデルは sampling.py と同じトライグラム（訓練データで数えた、直前の
2 文字から次の文字を予測する表。平滑化なし）。検証データの各位置で、
直前の 2 文字の文脈について、top-p（p = 0.5, 0.8, 0.95）が残す候補の
数と、top-k（k = 5）が残す確率の合計を求める。文脈が訓練データに一度も
出てこない位置は、分布が決まらないので除く。乱数は使わない。
    cd code && python ch06/sweep_topp_count.py > ch06/out/sweep_topp_count.txt
出力: ch06/out/sweep_topp_cdf.csv（残す個数 x 以下の位置の割合）,
      ch06/out/sweep_topp_examples.csv（3 つの文脈の、上から足した合計）
"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
text = load_corpus()
train_text, val_text = train_val_split(text)

table = defaultdict(Counter)  # table[直前の 2 文字][次の文字] = 回数
for i in range(2, len(train_text)):
    table[train_text[i - 2:i]][train_text[i]] += 1


def cumulative(ctx):
    """文脈 ctx の次の文字の確率を大きい順に並べ、上から足した合計。"""
    c = np.sort(np.array(list(table[ctx].values()), dtype=float))[::-1]
    return np.cumsum(c) / c.sum()


PS, K = [0.5, 0.8, 0.95], 5
cache = {}
kept = {p: [] for p in PS}   # 位置ごとの、top-p が残す個数
mass = []                    # 位置ごとの、top-k が残す確率の合計
skipped = 0
for i in range(2, len(val_text)):
    ctx = val_text[i - 2:i]
    if ctx not in table:
        skipped += 1
        continue
    if ctx not in cache:
        cache[ctx] = cumulative(ctx)
    cum = cache[ctx]
    before = np.concatenate([[0.0], cum[:-1]])  # 自分より上の合計
    for p in PS:
        kept[p].append(int((before < p).sum()))
    mass.append(cum[min(K, len(cum)) - 1])
n_pos = len(mass)
total = n_pos + skipped
print(f"検証データの位置 {total:,} 個のうち、文脈が訓練データにない"
      f" {skipped:,} 個（{100 * skipped / total:.1f}%）を除いた")
for p in PS:
    q = np.percentile(kept[p], [10, 25, 50, 75, 90])
    print(f"top-p {p:<4}  残す個数  10% {q[0]:.0f}  25% {q[1]:.0f}"
          f"  中央 {q[2]:.0f}  75% {q[3]:.0f}  90% {q[4]:.0f}"
          f"  （1 個だけの位置 {100 * np.mean(np.array(kept[p]) == 1):.0f}%）")
q = np.percentile(mass, [10, 25, 50, 75, 90])
print(f"top-k {K}     残す確率の合計  10% {q[0]:.2f}  25% {q[1]:.2f}"
      f"  中央 {q[2]:.2f}  75% {q[3]:.2f}  90% {q[4]:.2f}")

xs = np.unique(np.round(np.logspace(0, 3, 61)).astype(int))
rows = ["x," + ",".join(f"p{int(100 * p)}" for p in PS)]
for x in xs:
    rows.append(f"{x}," + ",".join(
        f"{np.mean(np.array(kept[p]) <= x):.4f}" for p in PS))
(OUT / "sweep_topp_cdf.csv").write_text("\n".join(rows) + "\n",
                                        encoding="utf-8")
print("---")

EXAMPLES = ["いく", "先生", "私の"]  # 候補が少ない・中くらい・多い文脈
R = 120                      # 図に描く順位の数
rows = ["rank," + ",".join(f"c{j}" for j in range(len(EXAMPLES)))]
cums = [cumulative(c) for c in EXAMPLES]
for r in range(R):
    rows.append(f"{r + 1}," + ",".join(
        f"{cum[min(r, len(cum) - 1)]:.4f}" for cum in cums))
(OUT / "sweep_topp_examples.csv").write_text("\n".join(rows) + "\n",
                                             encoding="utf-8")
for c, cum in zip(EXAMPLES, cums):
    before = np.concatenate([[0.0], cum[:-1]])
    top = table[c].most_common(3)
    print(f"「{c}」 候補 {len(cum)} 種類"
          f"  top-p 0.8 で残す {(before < 0.8).sum()} 個"
          f"  top-k {K} で残す合計 {cum[min(K, len(cum)) - 1]:.3f}  上位: "
          + "  ".join(f"{ch} {n / sum(table[c].values()):.3f}"
                      for ch, n in top))
