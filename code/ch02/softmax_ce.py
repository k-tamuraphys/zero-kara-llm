"""ミニ課題: softmax と交差エントロピーをテンソル演算で書く。"""

import math

import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint


# [snippet: functions]
def softmax(z):
    """z: (N, V) のロジット。各行を確率分布に直す。"""
    e = torch.exp(z)  # (N, V)。要素ごとに e^z
    return e / e.sum(dim=-1, keepdim=True)  # (N, V) / (N, 1)


def cross_entropy(z, y):
    """z: (N, V) のロジット、y: (N,) の正解の番号。損失の平均を返す。"""
    N, V = z.shape
    assert y.shape == (N,), f"y の形が {tuple(y.shape)}"
    p = softmax(z)  # (N, V)
    p_correct = p[torch.arange(N), y]  # (N,)。各行から正解の列を取り出す
    return -torch.log(p_correct).mean()  # 数 1 個
# [end: functions]


def softmax_stable(z):
    """数値的に安定な softmax（演習の参考）。各行の最大値を引く。"""
    z = z - z.max(dim=-1, keepdim=True).values  # (N, V) - (N, 1)
    e = torch.exp(z)  # どの要素も 0 より大きく 1 以下
    return e / e.sum(dim=-1, keepdim=True)


# [snippet: small]
z = torch.tensor([[2.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0]])  # 形 (2, 3)。2 個の予測 × 3 個の候補
y = torch.tensor([0, 2])  # 正解は、1 個目が番号 0、2 個目が番号 2
print(softmax(z))
print(cross_entropy(z, y))  # (-log 0.6652 - log 0.3333) / 2
big = torch.tensor([[1000.0, 999.0, 998.0]])  # 大きなロジット
print(softmax(big))  # e^1000 があふれて nan（数でない値）になる
print(softmax_stable(big))  # 最大値を引く版は、正しく計算できる
# [end: small]

print("---")
# [snippet: check]
torch.manual_seed(0)
z = 2 * torch.randn(6, 10)  # 乱数のロジット。形 (6, 10)
y = torch.randint(0, 10, (6,))  # 0〜9 の乱数の正解。形 (6,)
print(torch.allclose(softmax(z), torch.softmax(z, dim=-1)))
print(cross_entropy(z, y).item(), F.cross_entropy(z, y).item())
# [end: check]

print("---")
# [snippet: real]
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")  # トークナイザ（第 7 章）
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
ids = torch.tensor(tok.encode("どうも仕方がない。"))  # トークン番号の列
print(ids, [tok.decode([i]) for i in ids])
x, y = ids[:-1], ids[1:]  # 入力と、1 つ先にずらした正解。どちらも形 (2,)
with torch.no_grad():
    logits, loss = model(x[None], y[None])  # 入力 (1, 2) → (1, 2, 4096)
z = logits[0]  # 形 (2, 4096)。2 個の予測 × 4,096 個の候補
p = softmax(z)  # 形 (2, 4096)。各行の和は 1
print(p[torch.arange(2), y])  # 正解のトークンに割り当てられた確率
print(tok.decode([p[1].argmax().item()]))  # 2 個目の予測で確率が最大のトークン
print(cross_entropy(z, y).item(), loss.item())  # 自作の損失と、モデルの損失
# [end: real]

print("---")
# [snippet: uniform]
z0 = torch.zeros(2, 4096)  # どの候補にも同じスコアを付けた予測
print(cross_entropy(z0, y).item(), math.log(4096))
# [end: uniform]

print("---")
# [snippet: backward]
logits, loss = model(x[None], y[None])  # 今度は、記録を取りながら計算する
loss.backward()  # 損失を、モデルの全パラメータで偏微分する
grads = [q.grad for q in model.parameters()]
# 勾配の付いたテンソルの数と、その要素数の合計
print(len(grads), sum(g.numel() for g in grads))
print(tuple(grads[0].shape))  # 1 個目の勾配の形。パラメータと同じ形
# [end: backward]

print("---")
# [snippet: grad]
z = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)  # 形 (1, 3)
y = torch.tensor([0])  # 正解は番号 0
loss = cross_entropy(z, y)
loss.backward()  # 損失を、ロジットの各成分で偏微分する
print(softmax(z.detach()))  # 確率 p
print(z.grad)  # 勾配。p から「正解の位置だけ 1」のベクトルを引いた形
# [end: grad]
