"""第 5 章: 第 4 章の MLP を、SGD・モーメンタム・Adam で学習させて比べる。

隠れ層 2 枚・幅 32 の MLP、渦巻きデータ 300 点、300 ステップ。
学習率を 0.0001 から 3 まで変えて、学習後の損失を調べる。
    cd code && python ch05/optim_mlp.py > ch05/out/optim_mlp.txt
"""
import math

import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, load_spiral

torch.set_num_threads(1)
X, Y, _, _ = load_spiral()
STEPS = 300
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]

# [snippet: makers]
MAKERS = {
    "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr),
    "momentum": lambda params, lr: torch.optim.SGD(params, lr=lr,
                                                   momentum=0.9),
    "adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
}
# [end: makers]


def train(name, lr):
    torch.manual_seed(0)
    model = DeepMLP(2, init="default")       # 第 4 章と同じ形・同じ初期化
    optimizer = MAKERS[name](model.parameters(), lr)
    curve = []
    for step in range(STEPS + 1):
        loss = F.cross_entropy(model(X), Y)
        curve.append(loss.item())
        if not math.isfinite(curve[-1]):
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    curve += [float("nan")] * (STEPS + 1 - len(curve))
    return curve


curves = {(name, lr): train(name, lr) for name in MAKERS for lr in LRS}

print(f"{STEPS} ステップ後の訓練損失（nan は発散）")
print("学習率      SGD   モーメンタム     Adam")
with open("ch05/out/optim_mlp_sweep.csv", "w") as f:
    f.write("lr,sgd,momentum,adam\n")
    for lr in LRS:
        vals = [curves[(name, lr)][-1] for name in MAKERS]
        print(f"{lr:<8g}" + "".join(f"{v:10.4f}" for v in vals))
        # 図では 1.5 より大きい値と nan を 1.5 にそろえる（図の上端）
        f.write(f"{lr:g}," + ",".join(
            f"{min(v, 1.5) if math.isfinite(v) else 1.5:.5f}" for v in vals)
            + "\n")

print()
best = {}
for name in MAKERS:
    finals = {lr: curves[(name, lr)][-1] for lr in LRS
              if math.isfinite(curves[(name, lr)][-1])}
    best[name] = min(finals, key=finals.get)
    curve = curves[(name, best[name])]
    reach = next((i for i, v in enumerate(curve) if v < 0.05), None)
    print(f"{name:<9s} いちばんよい学習率 {best[name]:<6g}"
          f" 損失 {curve[-1]:.4f}  0.05 を切ったステップ {reach}")

with open("ch05/out/optim_mlp_curves.csv", "w") as f:
    f.write("step,sgd,momentum,adam\n")
    for i in range(STEPS + 1):
        f.write(f"{i}," + ",".join(
            f"{curves[(name, best[name])][i]:.5f}" for name in MAKERS) + "\n")

# 5.7 節の図: Adam で学習率が小さすぎる・ちょうどよい・大きすぎるとき
with open("ch05/out/optim_mlp_lr.csv", "w") as f:
    f.write("step,small,good,large\n")
    for i in range(STEPS + 1):
        f.write(f"{i}," + ",".join(
            f"{min(curves[('adam', lr)][i], 3.0):.5f}"
            for lr in (1e-4, 1e-2, 0.3)) + "\n")
