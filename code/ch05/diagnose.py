"""第 5 章: 学習がうまくいかないときの確かめ方。

(1) 最初の損失が log(クラス数) に近いか。
(2) 小さな 1 バッチ（8 点）だけを覚え込ませられるか。
    正しいコードと、わざとバグを入れたコードで比べる。
    cd code && python ch05/diagnose.py > ch05/out/diagnose.txt
"""
import math

import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, load_spiral

torch.set_num_threads(1)
X, Y, _, _ = load_spiral()

torch.manual_seed(0)
model = DeepMLP(10, init="he", norm=True, residual=True)
print(f"学習前の損失 {F.cross_entropy(model(X), Y).item():.3f}"
      f"   log 3 = {math.log(3):.3f}")
print(f"（基盤モデルなら log 4096 = {math.log(4096):.3f}）")
print("---")

torch.manual_seed(0)
idx = torch.randperm(len(X))[:8]             # 8 点だけ取り出す
xb, yb = X[idx], Y[idx]


def loss_ok(model, x, y):
    return F.cross_entropy(model(x), y)      # ロジットをそのまま渡す


# [snippet: bug]
def loss_buggy(model, x, y):
    probs = torch.softmax(model(x), dim=1)   # バグ: 先に softmax をかけた
    return F.cross_entropy(probs, y)         # 中でもう一度 softmax がかかる
# [end: bug]


def overfit(loss_fn, steps=300):
    torch.manual_seed(0)
    model = DeepMLP(10, init="he", norm=True, residual=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    record = {}
    # [snippet: one-batch]
    for step in range(steps + 1):
        loss = loss_fn(model, xb, yb)        # いつも同じ 8 点
        if step in (0, 10, 50, 100, 300):
            record[step] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # [end: one-batch]
    acc = (model(xb).argmax(dim=1) == yb).float().mean().item()
    return record, acc


print("同じ 8 点だけで学習したときの損失（1 バッチ過学習テスト）")
print("ステップ          0      10      50     100     300   正解率")
for name, fn in (("正しいコード", loss_ok), ("バグ入り    ", loss_buggy)):
    record, acc = overfit(fn)
    print(name + "".join(f"{v:8.4f}" for v in record.values())
          + f"   {acc * 100:.0f}%")
floor = -math.log(math.e / (math.e + 2))
print(f"バグ入りの損失の下限 -log(e / (e + 2)) = {floor:.4f}")
