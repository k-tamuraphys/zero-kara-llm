"""第 6 章（図の追加）: Adam の学習率を変えると、勾配降下法で作る
バイグラムモデルの学習曲線がどう変わるかを測る小さなスイープ。

bigram_nn.py と同じモデル（ロジットの表 W、最初は全部 0）と同じ損失
（回数の表を使った全データの損失）で、学習率だけを 0.05, 0.2, 0.5, 2, 5
の 5 通りに変えて 100 回更新する。W の初期値が 0 なので、乱数は使わない。
学習率 0.5 の行は ch06/out/bigram_nn_log.csv と同じになる。
    cd code && python ch06/sweep_lr.py > ch06/out/sweep_lr.txt
検証損失は、訓練データにある組とない組に分けた値も表示する。
出力: ch06/out/sweep_lr.csv（列 step, train_<学習率>, val_<学習率>）
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)
text = load_corpus()
train_text, val_text = train_val_split(text)
chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}


def counts(s):
    """文章 s の組 (前の文字, 次の文字) の回数の表 (V, V)。"""
    t = torch.tensor([stoi[ch] for ch in s])
    x, y = t[:-1], t[1:]
    return torch.bincount(x * V + y, minlength=V * V).view(V, V).float()


C, Cv = counts(train_text), counts(val_text)  # 訓練と検証の回数の表


def table_loss(W, C):
    """回数の表 C にまとめたデータ全体の平均の損失（bigram_nn.py と同じ）。"""
    return -(C * F.log_softmax(W, dim=1)).sum() / C.sum()


LRS = [0.05, 0.2, 0.5, 2.0, 5.0]
curves = {}
for lr in LRS:
    W = torch.zeros(V, V, requires_grad=True)  # 毎回、同じ初期値から
    optimizer = torch.optim.Adam([W], lr=lr)   # 変えるのは学習率だけ
    tr, va = [], []
    for step in range(101):
        loss = table_loss(W, C)
        with torch.no_grad():
            va.append(table_loss(W, Cv).item())
        tr.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    curves[lr] = (tr, va)
    with torch.no_grad():   # 検証の組を、訓練データにある組とない組に分ける
        lp, un = F.log_softmax(W, dim=1), (C == 0).float()
        unseen = -(Cv * un * lp).sum() / (Cv * un).sum()
        seen = -(Cv * (1 - un) * lp).sum() / (Cv * (1 - un)).sum()
    print(f"学習率 {lr:4}  100 回後  訓練 {tr[-1]:.4f}  検証 {va[-1]:.4f}"
          f"  （検証の最小 {min(va):.4f}、{va.index(min(va))} 回目）"
          f"  検証のうち 訓練にある組 {seen:.3f}  ない組 {unseen:.2f}")

head = ["step"] + [f"train_{lr:g}" for lr in LRS] + [f"val_{lr:g}"
                                                     for lr in LRS]
rows = [",".join(head)]
for s in range(101):
    rows.append(",".join([str(s)] + [f"{curves[lr][0][s]:.4f}" for lr in LRS]
                         + [f"{curves[lr][1][s]:.4f}" for lr in LRS]))
(OUT / "sweep_lr.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
