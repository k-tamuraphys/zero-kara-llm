"""第 6 章（図の追加）: 訓練データの量を変えると、n-グラムの検証損失が
どう変わるかを測る小さなスイープ。

訓練用の文章を、順に 64 個の区間に切る。データの量 1/m（m = 16, 8, 4,
2, 1）では、番号を m で割った余りが r の区間だけを使う（r = 0, 1, 2 の
3 通り。m = 2 は 2 通り、m = 1 は全部の 1 通り）。区間ごとに数えて
足すので、区間のつなぎ目に、ありもしない並びはできない。
検証データ（末尾 5%）と alpha の候補（ngram.py と同じ）は固定し、
n と量の組ごとに、検証損失が最小になる alpha を選ぶ。
全部の量（m = 1）の値は ngram.py と同じになる。
    cd code && python ch06/sweep_ngram_data.py > ch06/out/sweep_ngram_data.txt
出力: ch06/out/sweep_ngram_data.csv（列 n, m, chars, mean, lo, hi）,
      sweep_ngram_data_byn.csv（n ごとの 1 行。列 m16, lo16, hi16, ...）,
      sweep_ngram_data_bym.csv（量ごとの 1 行。列 n1, ..., n5）,
      sweep_ngram_data_best.csv（量ごとに、検証損失が最小の n）
"""
from collections import Counter
from pathlib import Path

import numpy as np

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
text = load_corpus()
train_text, val_text = train_val_split(text)
V = len(set(text))
ALPHAS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]  # ngram.py と同じ
K = 64                                       # 訓練用の文章を切る区間の数
cut = np.linspace(0, len(train_text), K + 1).astype(int)
blocks = [train_text[cut[k]:cut[k + 1]] for k in range(K)]


def count(parts, n):
    """区間ごとに n 文字の並びを数えて足す。文脈の回数も返す。"""
    grams = Counter()
    for s in parts:
        grams.update(s[i - n + 1:i + 1] for i in range(n - 1, len(s)))
    ctx = Counter()
    for g, c in grams.items():
        ctx[g[:-1]] += c                     # 文脈（前の n-1 文字）の回数
    return grams, ctx


def val_loss(grams, ctx, n):
    """検証データの損失を、alpha の候補ごとに計算し、最小のものを返す。"""
    pos = range(n - 1, len(val_text))
    cn = np.array([grams.get(val_text[i - n + 1:i + 1], 0) for i in pos])
    cc = np.array([ctx.get(val_text[i - n + 1:i], 0) for i in pos])
    best = min((-np.log((cn + a) / (cc + a * V)).mean(), a) for a in ALPHAS)
    return best  # (損失, alpha)


MS = [16, 8, 4, 2, 1]
res = {}                                     # res[(n, m)] = (平均, 最小, 最大)
rows = ["n,m,chars,mean,lo,hi"]
print(" n   量     字数       検証損失（3 通りの最小〜最大）  alpha")
for n in range(1, 6):
    for m in MS:
        losses, alphas, sizes = [], [], []
        for r in range(min(3, m)):         # 区間の選び方（余り r）
            parts = [b for k, b in enumerate(blocks) if k % m == r]
            grams, ctx = count(parts, n)
            loss, a = val_loss(grams, ctx, n)
            losses.append(loss)
            alphas.append(a)
            sizes.append(sum(len(b) for b in parts))
        mean = float(np.mean(losses))
        print(f"{n:2d}  1/{m:<2d}  {int(np.mean(sizes)):9,}  {mean:7.3f}"
              f"（{min(losses):.3f}〜{max(losses):.3f}）  {alphas}")
        rows.append(f"{n},{m},{int(np.mean(sizes))},{mean:.4f},"
                    f"{min(losses):.4f},{max(losses):.4f}")
        res[(n, m)] = (mean, min(losses), max(losses), int(np.mean(sizes)))


def save(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


save("sweep_ngram_data.csv", rows)
save("sweep_ngram_data_byn.csv", ["n," + ",".join(
    f"m{m},lo{m},hi{m}" for m in MS)] + [f"{n}," + ",".join(
        "{:.4f},{:.4f},{:.4f}".format(*res[(n, m)][:3]) for m in MS)
        for n in range(1, 6)])
save("sweep_ngram_data_bym.csv", ["m,chars," + ",".join(
    f"n{n},lo{n},hi{n}" for n in range(1, 6))] + [f"{m},{res[(1, m)][3]},"
    + ",".join("{:.4f},{:.4f},{:.4f}".format(*res[(n, m)][:3])
               for n in range(1, 6)) for m in MS])
best = [min(range(1, 6), key=lambda n: res[(n, m)][0]) for m in MS]
save("sweep_ngram_data_best.csv", ["m,chars,n,loss"] + [
    f"{m},{res[(n, m)][3]},{n},{res[(n, m)][0]:.4f}"
    for m, n in zip(MS, best)])
print("量ごとの最小: "
      + "  ".join(f"1/{m} は n={n}" for m, n in zip(MS, best)))
print("---")

# 例:「吾輩は猫であ」の次を当てるとき、n ごとに見る文脈と、その回数
ex, nxt = "吾輩は猫であ", "る"
print("文脈      文脈の回数  次が「る」  次の文字の種類")
for n in range(1, 6):
    grams, ctx = count([train_text], n)
    c = ex[len(ex) - n + 1:] if n > 1 else ""
    kinds = sum(1 for g in grams if g[:-1] == c)
    print(f"n={n} 「{c}」  {ctx.get(c, 0):9,}  {grams.get(c + nxt, 0):8,}"
          f"  {kinds:6,}")
