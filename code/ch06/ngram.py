"""第 6 章: 文脈を長くした n-グラムモデル（n = 1〜5）の損失と生成。

実行: cd code && python ch06/ngram.py
出力: 標準出力（"---" で区切る）, ch06/out/ngram.csv
"""
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

text = load_corpus()
train_text, val_text = train_val_split(text)
V = len(set(text))
ALPHAS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


# [snippet: ngram]
def count_ngrams(s, n):
    """文脈（直前の n-1 文字）ごとに、次の文字の回数を数える。"""
    table = defaultdict(Counter)
    for i in range(n - 1, len(s)):
        table[s[i - n + 1:i]][s[i]] += 1  # table[文脈][次の文字]
    return table


def loss_of(table, s, n, alpha):
    """加算平滑化した n-グラムモデルで、文章 s を予測したときの損失。"""
    total = 0.0
    for i in range(n - 1, len(s)):
        row = table.get(s[i - n + 1:i], {})  # 見たことのない文脈なら空
        p = (row.get(s[i], 0) + alpha) / (sum(row.values()) + alpha * V)
        total -= math.log(p)
    return total / (len(s) - n + 1)
# [end: ngram]


def train_loss(table):
    """平滑化なし（alpha = 0）の訓練損失。表だけから計算できる。"""
    total, count = 0.0, 0
    for row in table.values():
        m = sum(row.values())
        total -= sum(c * math.log(c / m) for c in row.values())
        count += m
    return total / count


tables = {}
rows = ["n,kinds,unseen,train,val,alpha"]
print(" n   マス目の数   現れた並び   未観測   訓練損失  検証損失（alpha）")
for n in range(1, 6):
    table = tables[n] = count_ngrams(train_text, n)
    kinds = sum(len(row) for row in table.values())
    unseen = sum(val_text[i] not in table.get(val_text[i - n + 1:i], {})
                 for i in range(n - 1, len(val_text)))
    unseen /= len(val_text) - n + 1
    val, alpha = min((loss_of(table, val_text, n, a), a) for a in ALPHAS)
    tr = train_loss(table)
    print(f"{n:2d}  {float(V) ** n:12.1e}  {kinds:11,}  {100 * unseen:5.1f}%"
          f"  {tr:8.3f}  {val:8.3f}（{alpha}）")
    rows.append(f"{n},{kinds},{unseen:.4f},{tr:.4f},{val:.4f},{alpha}")
(OUT / "ngram.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("---")

# n ごとの生成例（平滑化なし。続いたことのある文字を、回数に比例して引く）
for n in range(1, 6):
    rng = random.Random(0)
    out = "\n　吾輩は"
    for _ in range(30):
        row = tables[n][out[len(out) - n + 1:]]
        out += rng.choices(list(row), weights=list(row.values()))[0]
    print(f"n={n}  " + out[2:].replace("\n", "／"))
