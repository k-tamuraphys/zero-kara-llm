"""第 5 章（追加のスイープ）: 学習率を変えて、深い MLP の部品を比べる。

隠れ層 20 枚・幅 32 の MLP で、SGD の学習率を 0.01〜1 の 5 通りに変え、
「He」「He＋レイヤー正規化」「He＋レイヤー正規化＋残差接続」を比べる。
1,000 ステップ、乱数の種 0〜2 の 3 回で、学習できた回数を数える。
    cd code && OMP_NUM_THREADS=2 python ch05/sweep_lr_depth.py \
        > ch05/out/sweep_lr_depth.txt
csv は ch05/out/sweep_lr_depth.csv に書く。
"""
import math

import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, load_spiral

torch.set_num_threads(1)
X, Y, _, _ = load_spiral()
STEPS = 1000
SEEDS = (0, 1, 2)


def train(model, lr):
    """全データ（300 点）の勾配で STEPS 回更新し、損失の列を返す。"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    curve = []
    for step in range(STEPS):
        loss = F.cross_entropy(model(X), Y)
        curve.append(loss.item())
        if not math.isfinite(curve[-1]):     # 発散したら打ち切る
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return curve


@torch.no_grad()
def accuracy(model):
    return (model(X).argmax(dim=1) == Y).float().mean().item()


# 学習率を変える（隠れ層 20 枚）
LRS = (0.01, 0.03, 0.1, 0.3, 1.0)
CONFIGS = {
    "he": dict(init="he"),
    "he_ln": dict(init="he", norm=True),
    "he_res_ln": dict(init="he", norm=True, residual=True),
}
print("隠れ層 20 枚、SGD、1,000 ステップ。3 回のうち学習できた回数")
print("学習率   He   He＋正規化   He＋正規化＋残差")
rows = []
for lr in LRS:
    counts = []
    for key, kw in CONFIGS.items():
        ok = 0
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = DeepMLP(20, **kw)
            curve = train(model, lr)
            ok += int(len(curve) == STEPS and math.isfinite(curve[-1])
                      and accuracy(model) >= 0.99)
        counts.append(ok)
    rows.append((lr, counts))
    print(f"  {lr:<5g}" + "".join(f"{c:8d} 回" for c in counts))
with open("ch05/out/sweep_lr_depth.csv", "w") as f:
    f.write("lr," + ",".join(CONFIGS) + "\n")
    for lr, counts in rows:
        f.write(f"{lr}," + ",".join(str(c) for c in counts) + "\n")
