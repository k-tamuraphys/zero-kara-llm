"""第 6 章: 数えるだけで作る、文字単位のバイグラム言語モデル。

実行: cd code && python ch06/bigram_count.py
      （bash ch06/run_all.sh が出力を ch06/out/ に保存する）
出力: 標準出力（"---" で区切る）, ch06/out/bigram_alpha.csv, bigram_heat.csv
"""
import math
from pathlib import Path

import torch

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(0)

# [snippet: data]
text = load_corpus()                          # 漱石の 14 作品、約 241 万字
train_text, val_text = train_val_split(text)  # 末尾の 5% は検証用に取っておく
chars = sorted(set(text))                     # 出てくる文字の一覧 = 語彙
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # 文字 → 番号


def encode(s):
    return torch.tensor([stoi[ch] for ch in s])


train, val = encode(train_text), encode(val_text)
x, y = train[:-1], train[1:]                  # 入力と正解。1 つずらすだけ
# [end: data]
print(f"語彙 V = {V}  訓練 {len(train):,} 字  検証 {len(val):,} 字")
print("入力 x:", " ".join(chars[i] for i in x[3:11]))
print("正解 y:", " ".join(chars[i] for i in y[3:11]))
print("---")

# [snippet: count]
# 組 (a, b) に通し番号 a*V+b を振り、番号ごとの出現回数をまとめて数える
C = torch.bincount(x * V + y, minlength=V * V).view(V, V)  # (V, V)
print(f"組の総数 {C.sum().item():,}  表のマス目 {V * V:,}"
      f"  1 回以上現れた組 {(C > 0).sum().item():,}")

# 各行を、その行の合計で割る。行 a は「a の次の文字」の確率分布になる
# （訓練データにない文字の行は合計が 0。0 で割らないよう 1 にしておく）
P = C / C.sum(dim=1, keepdim=True).clamp(min=1)            # (V, V)
# [end: count]


def followers(ch, k=5):
    """文字 ch の次に来やすい文字を、確率の高い順に k 個表示する。"""
    prob, idx = torch.topk(P[stoi[ch]], k)
    body = "  ".join(f"{chars[i]} {q:.3f}" for q, i in zip(prob, idx))
    print(f"{ch}（{C[stoi[ch]].sum().item():,} 回）→ {body}")


for ch in "っで吾の":
    followers(ch)
print("---")


# [snippet: loss]
def loss_of(P, data):
    """確率の表 P で、data の「次の文字」をすべて予測したときの損失。"""
    a, b = data[:-1], data[1:]
    return -torch.log(P[a, b]).mean().item()  # 正解に付けた確率の -log の平均
# [end: loss]


print(f"訓練データの損失 {loss_of(P, train):.4f}")
print(f"検証データの損失 {loss_of(P, val):.4f}")
unseen = (C[val[:-1], val[1:]] == 0).sum().item()
print(f"検証データの組 {len(val) - 1:,} 個のうち、"
      f"訓練データにない組 {unseen:,} 個"
      f"（{100 * unseen / (len(val) - 1):.2f}%）")
print("---")


# [snippet: smooth]
def smooth(C, alpha):
    """全部のマス目に alpha 回ぶんを足してから、確率に直す（加算平滑化）。"""
    return (C + alpha) / (C.sum(dim=1, keepdim=True) + alpha * V)
# [end: smooth]


for alpha in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    Pa = smooth(C, alpha)
    print(f"alpha = {alpha:<5}  訓練 {loss_of(Pa, train):.4f}"
          f"  検証 {loss_of(Pa, val):.4f}")
rows = []
for alpha in [0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05,
              0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]:
    Pa = smooth(C, alpha)
    rows.append(f"{alpha},{loss_of(Pa, train):.4f},{loss_of(Pa, val):.4f}")
(OUT / "bigram_alpha.csv").write_text(
    "alpha,train,val\n" + "\n".join(rows) + "\n", encoding="utf-8")
print("---")

# 基準線: 一様分布、ユニグラム（前の文字を見ない）、バイグラム
ALPHA = 0.01                                   # 検証損失が最小だった値
n1 = torch.bincount(train, minlength=V)        # 文字ごとの出現回数 (V,)
p1 = (n1 + ALPHA) / (n1.sum() + ALPHA * V)
uni = -torch.log(p1[val]).mean().item()
Pb = smooth(C, ALPHA)
big = loss_of(Pb, val)
print(f"一様分布    損失 {math.log(V):.3f}  パープレキシティ {V:7.1f}")
print(f"ユニグラム  損失 {uni:.3f}  パープレキシティ {math.exp(uni):7.1f}")
print(f"バイグラム  損失 {big:.3f}  パープレキシティ {math.exp(big):7.1f}")
print("---")


# [snippet: score]
def log_prob(sentence):
    """「。」の直後に sentence が続く確率の対数（バイグラム、alpha=0.01）。"""
    ids = encode("。" + sentence)
    return torch.log(Pb[ids[:-1], ids[1:]]).sum().item()


for s in ["吾輩は猫である。", "猫は吾輩である。", "輩猫はで吾ある。"]:
    lp = log_prob(s)
    print(f"{s}  log p = {lp:7.2f}   p = {math.exp(lp):.1e}")
# [end: score]
print("---")

# [snippet: generate]
g = torch.Generator().manual_seed(0)  # 乱数の種を固定する
ix = stoi["。"]                       # 「。」の次の文字から書き始める
out = []
for _ in range(68):
    # P[ix] は「ix の次の文字」の確率分布 (V,)。それに従って 1 個引く
    ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
    out.append(chars[ix])
# [end: generate]
sample = "".join(out).replace("\n", "／")  # 改行は「／」で表示する
print(sample[:34])
print(sample[34:])

# 図（確率の表のヒートマップ）のデータ
heat = "っしなあでこたているの。"
rows = ["i,j,p"]
for i, a in enumerate(heat):
    for j, b in enumerate(heat):
        rows.append(f"{i},{j},{P[stoi[a], stoi[b]].item():.3f}")
(OUT / "bigram_heat.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
