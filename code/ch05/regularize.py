"""第 5 章: 過学習を、重み減衰・ドロップアウト・早期終了で抑える。

渦巻きデータ 300 点のうち約 15% のラベルをわざと間違えておく。
幅 128・隠れ層 2 枚の MLP を AdamW で 2,000 ステップ学習させる。
    cd code && python ch05/regularize.py > ch05/out/regularize.txt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ch04.data import make_spiral

torch.set_num_threads(1)


# [snippet: dropout]
def dropout(h, p, training):
    """学習中だけ、各成分を確率 p で 0 にし、残りを 1/(1-p) 倍する。"""
    if not training or p == 0.0:
        return h                             # 評価のときは何もしない
    keep = (torch.rand_like(h) >= p).float() # 残す成分が 1、消す成分が 0
    return h * keep / (1.0 - p)              # 平均が変わらないように拡大
# [end: dropout]


torch.manual_seed(0)
h = torch.ones(2, 8)
print("入力           :", h[0].tolist())
print("学習中 (p=0.5) :", dropout(h, 0.5, True)[0].tolist())
print("評価時         :", dropout(h, 0.5, False)[0].tolist())
big = dropout(torch.ones(100000), 0.5, True)
print(f"10 万個の平均   : {big.mean():.3f}"
      f"（0 になった割合 {(big == 0).float().mean():.3f}）")
print("---")


def tensors(seed):
    X, Y = make_spiral(n_per_class=100, seed=seed)
    return torch.tensor(X), torch.tensor(Y)


X, Y_clean = tensors(0)                      # 訓練データ
Xv, Yv = tensors(1)                          # 検証データ（ラベルは正しい）
Xt, Yt = tensors(2)                          # テストデータ（ラベルは正しい）
g = torch.Generator().manual_seed(0)
flip = torch.rand(len(Y_clean), generator=g) < 0.15
shift = torch.randint(1, 3, (len(Y_clean),), generator=g)
Y = torch.where(flip, (Y_clean + shift) % 3, Y_clean)   # 間違えたラベル
print(f"訓練データ {len(Y)} 点のうち、ラベルを間違えた数: {flip.sum().item()}")


# [snippet: model]
class MLP(nn.Module):
    def __init__(self, width=128, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(2, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, 3)
        self.drop = nn.Dropout(p)            # 自作の dropout と同じ働き

    def forward(self, x):
        h = self.drop(torch.relu(self.fc1(x)))
        h = self.drop(torch.relu(self.fc2(h)))
        return self.out(h)
# [end: model]


@torch.no_grad()
def measure(model, X, Y):
    model.eval()                             # 評価モード: ドロップアウトを切る
    logits = model(X)
    model.train()                            # 学習モードに戻す
    return (F.cross_entropy(logits, Y).item(),
            (logits.argmax(dim=1) == Y).float().mean().item())


def train(p=0.0, weight_decay=0.0, steps=2000):
    torch.manual_seed(0)
    model = MLP(p=p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01,
                                  weight_decay=weight_decay)
    history = []                             # (ステップ, 訓練, 検証, テスト)
    for step in range(steps + 1):
        if step % 50 == 0:
            history.append((step, measure(model, X, Y),
                            measure(model, Xv, Yv), measure(model, Xt, Yt)))
        loss = F.cross_entropy(model(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return history


def pad(s, width):
    return s + " " * (width - sum(2 if ord(c) > 255 else 1 for c in s))


def show(name, record):
    step, tr, va, te = record
    print(pad(name, 26) + f"{tr[0]:6.3f} {tr[1] * 100:5.1f}%"
          f" {va[0]:8.3f} {te[0]:8.3f} {te[1] * 100:6.1f}%")


runs = {
    "対策なし": train(),
    "重み減衰 0.1": train(weight_decay=0.1),
    "重み減衰 1.0": train(weight_decay=1.0),
    "ドロップアウト 0.2": train(p=0.2),
    "ドロップアウト 0.5": train(p=0.5),
}
print("---")
print("2,000 ステップ後（訓練はラベルを間違えたままのデータで測る）")
print(pad("設定", 26) + "訓練損失 正解率 検証損失 テスト損失 正解率")
for name, history in runs.items():
    show(name, history[-1])
best = min(runs["対策なし"], key=lambda r: r[2][0])   # 検証損失が最小の時点
show(f"早期終了（ステップ {best[0]}）", best)

with open("ch05/out/regularize.csv", "w") as f:
    f.write("step,train,val,train_wd,val_wd,train_drop,val_drop\n")
    keys = ("対策なし", "重み減衰 1.0", "ドロップアウト 0.5")
    for i in range(len(runs["対策なし"])):
        row = [runs["対策なし"][i][0]]
        for k in keys:
            row += [runs[k][i][1][0], runs[k][i][2][0]]
        f.write(",".join(f"{v:.5g}" for v in row) + "\n")
