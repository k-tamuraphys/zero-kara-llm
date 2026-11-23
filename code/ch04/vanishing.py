"""層を深くすると学習が難しくなることを確かめる（勾配消失・勾配爆発の予告）。

(1) 隠れ層の枚数と活性化関数を変えて、渦巻きデータを同じ設定で学習させる。
(2) 隠れ層 10 枚の MLP で、学習前の勾配の大きさを層ごとに測る。
(3) ReLU の深い MLP を、初期値と学習率を変えてもう一度学習させる。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_spiral

torch.set_num_threads(2)
X, Y = make_spiral(n_per_class=100, seed=0)
X, Y = torch.tensor(X), torch.tensor(Y)
ACTS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "ReLU": torch.relu}


# [snippet: deep]
class DeepMLP(nn.Module):
    """幅 32 の隠れ層を depth 枚重ねた MLP。"""

    def __init__(self, depth, act, width=32, std=None):
        super().__init__()
        sizes = [2] + [width] * depth
        self.hidden = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(depth)])
        self.out = nn.Linear(width, 3)
        self.act = act
        if std is not None:  # 重みの初期値の大きさを指定する実験用
            for layer in self.hidden:
                nn.init.normal_(layer.weight, mean=0.0, std=std)

    def forward(self, x, keep=None):
        for layer in self.hidden:
            x = self.act(layer(x))
            if keep is not None:  # 途中の出力の勾配も後で見られるようにする
                x.retain_grad()
                keep.append(x)
        return self.out(x)
# [end: deep]


def train(depth, act, lr=0.5, std=None):
    """2,000 ステップ学習させ、最後の訓練損失と正解率を文字列で返す。"""
    torch.manual_seed(0)
    model = DeepMLP(depth, act, std=std)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(2001):
        loss = F.cross_entropy(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == Y).float().mean().item()
    return f"{loss.item():.3f} ({acc:4.0%})"


# --- (1) 深さを変えて学習
print("隠れ層の枚数  " + "".join(f"{name:>16s}" for name in ACTS))
for depth in (2, 5, 10, 20):
    row = f"{depth:6d}      "
    for name, act in ACTS.items():
        row += "   " + train(depth, act)
    print(row)


# --- (2) 学習前の勾配の大きさを層ごとに測る
def grad_norms(act, std=None, depth=10):
    torch.manual_seed(0)
    model = DeepMLP(depth, act, std=std)
    keep = []
    loss = F.cross_entropy(model(X, keep), Y)
    loss.backward()
    return [h.grad.norm().item() for h in keep]  # 各隠れ層の出力での勾配


cols = {
    "sigmoid": grad_norms(torch.sigmoid),
    "relu": grad_norms(torch.relu),
    "relu_big": grad_norms(torch.relu, std=1.0),
    "relu_good": grad_norms(torch.relu, std=0.25),
}
with open("ch04/out/grad_norms.csv", "w") as f:
    f.write("layer," + ",".join(cols) + "\n")
    for i in range(10):
        f.write(f"{i + 1}," + ",".join(f"{v[i]:.4e}" for v in cols.values())
                + "\n")
print()
print("隠れ層 10 枚、学習前。各層の出力での勾配の大きさ（第 1 層 / 第 10 層）")
for name, v in cols.items():
    print(f"  {name:10s} {v[0]:.1e} / {v[-1]:.1e}")

# --- (3) ReLU の深い MLP を、初期値と学習率を変えて学習し直す
print()
print("ReLU。初期値と学習率を変えて学習（学習後の訓練損失と正解率）")
for std, lr in ((0.25, 0.5), (None, 0.1), (0.25, 0.1)):
    init = "標準の初期値" if std is None else f"標準偏差 {std}"
    row = f"  {init}、学習率 {lr}:"
    for depth in (10, 20):
        row += f"  {depth} 枚 " + train(depth, torch.relu, lr=lr, std=std)
    print(row)
