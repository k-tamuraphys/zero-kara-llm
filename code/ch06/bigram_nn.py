"""第 6 章: 同じバイグラムモデルを、勾配降下で学習して作り直す。

実行: cd code && python ch06/bigram_nn.py
出力: 標準出力（"---" で区切る）,
      ch06/out/bigram_nn_log.csv, bigram_nn_scatter.csv
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(0)

text = load_corpus()
train_text, val_text = train_val_split(text)
chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
train = torch.tensor([stoi[ch] for ch in train_text])
val = torch.tensor([stoi[ch] for ch in val_text])
x, y = train[:-1], train[1:]                  # 入力と正解 (約 229 万,)
xv, yv = val[:-1], val[1:]


# [snippet: model]
class BigramLM(nn.Module):
    """前の文字の番号から、次の文字のロジットを表から引くだけのモデル。"""

    def __init__(self, V):
        super().__init__()
        # 行 = 前の文字、列 = 次の文字。最初は全部 0（どの文字も同じ確率）
        self.W = nn.Parameter(torch.zeros(V, V))  # (V, V)

    def forward(self, x):   # x: (B,) 前の文字の番号
        return self.W[x]    # (B, V) 番号 x の行を取り出す = ロジット
# [end: model]


model = BigramLM(V)
logits = model(x[:5])                         # 最初の 5 個の予測
print("ロジットの形", tuple(logits.shape))
print(f"学習前の損失 {F.cross_entropy(logits, y[:5]).item():.4f}")
print("---")


# [snippet: table-loss]
def table_loss(model, C):
    """数の表 C にまとめたデータ全体の、平均の損失。"""
    logp = F.log_softmax(model.W, dim=1)      # (V, V) 各行が log p(b|a)
    return -(C * logp).sum() / C.sum()        # 組 (a,b) の損失を C[a,b] 回足す


C = torch.bincount(x * V + y, minlength=V * V).view(V, V).float()
# [end: table-loss]
a, b = x[:20000], y[:20000]                   # 確かめ: 最初の 2 万組で比べる
C_small = torch.bincount(a * V + b, minlength=V * V).view(V, V).float()
torch.manual_seed(0)
check = BigramLM(V)
check.W.data = torch.randn(V, V)              # でたらめな表で確かめる
print(f"1 組ずつ {F.cross_entropy(check(a), b).item():.3f}"
      f"   数の表 {table_loss(check, C_small).item():.3f}")
del check, C_small
print("---")


@torch.no_grad()
def pair_loss(model, x, y, chunk=20000):
    """1 組ずつの普通の計算で、平均の損失を求める（小分けにして計算）。"""
    total = 0.0
    for i in range(0, len(x), chunk):
        logits = model(x[i:i + chunk])
        total += F.cross_entropy(logits, y[i:i + chunk],
                                 reduction="sum").item()
    return total / len(x)


# [snippet: train]
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
for step in range(101):
    loss = table_loss(model, C)               # 訓練データ全体の損失
    if step % 20 == 0:
        print(f"step {step:3d}  訓練 {loss.item():.4f}"
              f"  検証 {pair_loss(model, xv, yv):.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# [end: train]
print("---")

# 学習曲線の記録を取り直す（毎ステップ）。上と同じ計算なので結果は同じ
torch.manual_seed(0)
model = BigramLM(V)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
Cv = torch.bincount(xv * V + yv, minlength=V * V).view(V, V).float()
t0 = time.time()
rows = ["step,train,val"]
for step in range(101):
    loss = table_loss(model, C)
    with torch.no_grad():
        v = table_loss(model, Cv).item()
        rows.append(f"{step},{loss.item():.4f},{v:.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
(OUT / "bigram_nn_log.csv").write_text("\n".join(rows) + "\n",
                                       encoding="utf-8")
elapsed = time.time() - t0

# [snippet: compare]
P_count = C / C.sum(dim=1, keepdim=True).clamp(min=1)  # 数えて作った確率の表
P_model = torch.softmax(model.W.detach(), dim=1)       # 学習した表を確率に直す
for ch in "っで吾":
    for name, P in [("数えた", P_count), ("学習した", P_model)]:
        prob, idx = torch.topk(P[stoi[ch]], 4)
        body = "  ".join(f"{chars[i]} {q:.3f}" for q, i in zip(prob, idx))
        print(f"{ch} → {body}   （{name}）")
# [end: compare]
print("---")
count_loss = -(C * torch.log(P_count.clamp(min=1e-30))).sum().item()
count_loss /= C.sum().item()
with torch.no_grad():
    final = table_loss(model, C).item()
print(f"数えて作った表の訓練損失 {count_loss:.4f}")
print(f"学習した表の訓練損失     {final:.4f}")
print(f"学習の時間 {elapsed:.0f} 秒（101 回）")

# 散布図のデータ: 1 回以上現れた組から 400 個を選び、二つの確率を並べる
g = torch.Generator().manual_seed(0)
idx = torch.nonzero(C.flatten() > 0).flatten()
pick = idx[torch.randperm(len(idx), generator=g)[:400]]
rows = ["count,p_count,p_model"]
for k in pick.tolist():
    i, j = divmod(k, V)
    rows.append(f"{int(C[i, j])},{P_count[i, j].item():.6g},"
                f"{P_model[i, j].item():.6g}")
(OUT / "bigram_nn_scatter.csv").write_text("\n".join(rows) + "\n",
                                           encoding="utf-8")
