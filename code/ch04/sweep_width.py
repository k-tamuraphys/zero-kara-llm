"""隠れ層の幅を変えて、渦巻きデータの学習がどう変わるかを見る。

mlp_torch.py と同じ設定（隠れ層 2 枚、SGD、学習率 0.5、1,000 回の更新、
訓練 300 点・テスト 300 点）で、幅だけを 6 通りに変える。
乱数の種（初期値）は 3 通り。種 0・幅 32 が mlp_torch.py と同じ実験になる。
出力: ch04/out/sweep_width.csv（幅・種ごとの正解率）、
      ch04/out/sweep_width_regions.csv（種 0 での予測の領域）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_spiral

torch.set_num_threads(2)
WIDTHS = (2, 4, 8, 16, 32, 64)
SEEDS = (0, 1, 2)


class MLP(nn.Module):
    """mlp_torch.py の MLP と同じ形（隠れ層 2 枚）。"""

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

    def forward(self, x):  # x: (N, n_in)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)  # (N, n_out) ロジット


def accuracy(model, X, Y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == Y).float().mean().item()


X, Y = make_spiral(n_per_class=100, seed=0)  # 訓練 300 点
X, Y = torch.tensor(X), torch.tensor(Y)
Xt, Yt = make_spiral(n_per_class=100, seed=1)  # テスト 300 点
Xt, Yt = torch.tensor(Xt), torch.tensor(Yt)

# 予測の領域を塗るための格子（mlp_torch.py と同じ 60 × 60）
n = 60
g = torch.linspace(-1.15, 1.15, n)
grid = torch.stack([g.repeat(n), g.repeat_interleave(n)], dim=1)

rows, regions = [], {}
for width in WIDTHS:
    for seed in SEEDS:
        torch.manual_seed(seed)  # 種で初期値だけが変わる
        model = MLP(2, width, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        for step in range(1000):  # 1,000 回更新する
            loss = F.cross_entropy(model(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            tr_loss = F.cross_entropy(model(X), Y).item()
        n_params = sum(p.numel() for p in model.parameters())
        tr, te = accuracy(model, X, Y), accuracy(model, Xt, Yt)
        rows.append((width, seed, n_params, tr_loss, tr, te))
        print(f"幅 {width:2d}  種 {seed}  パラメータ {n_params:5d}  "
              f"訓練損失 {tr_loss:.4f}  訓練 {tr:.3f}  テスト {te:.3f}")
        if seed == 0:
            with torch.no_grad():
                regions[width] = model(grid).argmax(dim=1).tolist()

with open("ch04/out/sweep_width.csv", "w") as f:
    f.write("width,seed,params,train_loss,train_acc,test_acc\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.4f},{r[5]:.4f}\n")
with open("ch04/out/sweep_width_regions.csv", "w") as f:
    f.write("x1,x2," + ",".join(f"c{w}" for w in WIDTHS) + "\n")
    for i, (x1, x2) in enumerate(grid.tolist()):
        f.write(f"{x1:.4f},{x2:.4f},"
                + ",".join(str(regions[w][i]) for w in WIDTHS) + "\n")
