"""第 5 章（追加のスイープ）: 学習ループの側の設定を何通りかに変える。

(1) ウォームアップの長さ（schedule.py と同じ設定: 隠れ層 20 枚、Adam、
    最大学習率 0.1、ミニバッチ 32 件、400 ステップ、乱数の種 0〜2）。
(2) 勾配クリッピングの上限（clip.py と同じ設定: 隠れ層 10 枚、He、
    SGD 学習率 0.5、500 ステップ）。比べるために、クリッピングなしで
    学習率だけを下げた場合も走らせる。
(3) 重み減衰とドロップアウトの強さ（regularize.py と同じ設定）。
    cd code && OMP_NUM_THREADS=2 python ch05/sweep_loop.py \
        > ch05/out/sweep_loop.txt
"""
import math

import torch
import torch.nn.functional as F

from ch04.data import make_spiral
from ch05.deepmlp import DeepMLP, evaluate, load_spiral
from llmbook.train import lr_at

torch.set_num_threads(1)
X, Y, Xt, Yt = load_spiral()

# (1) ウォームアップの長さ
WARMUPS = (0, 5, 10, 20, 40, 80, 160)
print("(1) ウォームアップの長さ（3 回の平均）")
print("ウォームアップ  最初の 50 ステップの最大  最後の訓練損失")
with open("ch05/out/sweep_warmup.csv", "w") as f:
    f.write("warmup,peak,final,peak_min,peak_max\n")
    for warmup in WARMUPS:
        peaks, lasts = [], []
        for seed in range(3):
            torch.manual_seed(seed)
            model = DeepMLP(20, init="he", norm=True, residual=True)
            opt = torch.optim.Adam(model.parameters(), lr=0.1)
            losses = []
            for step in range(400):
                for group in opt.param_groups:
                    group["lr"] = lr_at(step, 0.1, warmup, 400, 0.1)
                idx = torch.randint(0, len(X), (32,))
                loss = F.cross_entropy(model(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            peaks.append(max(losses[:50]))
            lasts.append(evaluate(model, X, Y)[0])
        peak, last = sum(peaks) / 3, sum(lasts) / 3
        print(f"  {warmup:4d}          {peak:6.2f}              {last:.4f}"
              f"   （種ごとの最大: "
              + ", ".join(f"{v:.2f}" for v in peaks) + "）")
        f.write(f"{warmup},{peak:.4f},{last:.5f},{min(peaks):.4f},"
                f"{max(peaks):.4f}\n")
print("---")


# (2) クリッピングの上限と、学習率を下げた場合
#     clip.py の自作関数と同じもの（clip.py を読み込むと元の実験が走り、
#     出力を上書きするので、ここに写してある）
def clip_grad_norm(params, max_norm):
    params = list(params)
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    if total > max_norm:
        for p in params:
            p.grad *= max_norm / total
    return total.item()


def train_clip(max_norm, lr):
    torch.manual_seed(0)
    model = DeepMLP(10, init="he")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses, clipped = [], 0
    for step in range(500):
        loss = F.cross_entropy(model(X), Y)
        opt.zero_grad()
        loss.backward()
        norm = clip_grad_norm(model.parameters(), max_norm)
        clipped += int(norm > max_norm)
        opt.step()
        losses.append(loss.item())
    return max(losses), evaluate(model, X, Y)[0], clipped


print("(2) 隠れ層 10 枚、He、SGD、500 ステップ")
print("設定                       途中の最大  最後の訓練損失  縮めた回数")
with open("ch05/out/sweep_clip.csv", "w") as f:
    f.write("kind,max_norm,lr,peak,final,clipped\n")
    for max_norm in (0.25, 0.5, 1.0, 2.0, 4.0, math.inf):
        peak, last, n = train_clip(max_norm, 0.5)
        print(f"  学習率 0.5・上限 {max_norm:<5g}   {peak:8.2f}"
              f"   {last:12.4f}   {n:6d}")
        f.write(f"clip,{max_norm},0.5,{peak:.4f},{last:.5f},{n}\n")
    for lr in (0.25, 0.1, 0.05):
        peak, last, n = train_clip(math.inf, lr)
        print(f"  学習率 {lr:<4g}・なし        {peak:8.2f}"
              f"   {last:12.4f}   {n:6d}")
        f.write(f"lr,inf,{lr},{peak:.4f},{last:.5f},{n}\n")
print("---")

# (3) 重み減衰とドロップアウトの強さ（regularize.py と同じデータとモデル。
#     regularize.py を読み込むと元の実験が走り、出力を上書きするので、
#     同じ定義をここに写してある）
def tensors(seed):
    Xs, Ys = make_spiral(n_per_class=100, seed=seed)
    return torch.tensor(Xs), torch.tensor(Ys)


Xr, Yr_clean = tensors(0)                    # 訓練データ
Xv, Yv = tensors(1)                          # 検証データ（ラベルは正しい）
Xs, Ys = tensors(2)                          # テストデータ（ラベルは正しい）
g = torch.Generator().manual_seed(0)
flip = torch.rand(len(Yr_clean), generator=g) < 0.15
shift = torch.randint(1, 3, (len(Yr_clean),), generator=g)
Yr = torch.where(flip, (Yr_clean + shift) % 3, Yr_clean)


class MLP(torch.nn.Module):
    def __init__(self, width=128, p=0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.out = torch.nn.Linear(width, 3)
        self.drop = torch.nn.Dropout(p)

    def forward(self, x):
        h = self.drop(torch.relu(self.fc1(x)))
        h = self.drop(torch.relu(self.fc2(h)))
        return self.out(h)


@torch.no_grad()
def measure(model, Xm, Ym):
    model.eval()                             # ドロップアウトを切って測る
    logits = model(Xm)
    model.train()
    return (F.cross_entropy(logits, Ym).item(),
            (logits.argmax(dim=1) == Ym).float().mean().item())


def train_reg(p=0.0, weight_decay=0.0, steps=2000):
    torch.manual_seed(0)
    model = MLP(p=p)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01,
                            weight_decay=weight_decay)
    for step in range(steps):
        loss = F.cross_entropy(model(Xr), Yr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return measure(model, Xr, Yr), measure(model, Xv, Yv), \
        measure(model, Xs, Ys)


print("(3) 重み減衰とドロップアウトの強さ（2,000 ステップ後）")
print("設定               訓練正解率  検証損失  テスト損失  テスト正解率")
with open("ch05/out/sweep_reg.csv", "w") as f:
    f.write("kind,strength,train_acc,val_loss,test_loss,test_acc\n")
    for kind, values in (("wd", (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)),
                         ("drop", (0.0, 0.1, 0.2, 0.3, 0.5, 0.7))):
        for v in values:
            if kind == "wd":
                tr, va, te = train_reg(weight_decay=v)
            else:
                tr, va, te = train_reg(p=v)
            name = ("重み減衰" if kind == "wd" else "ドロップアウト")
            print(f"  {name} {v:<5g}  {tr[1] * 100:8.1f}%  {va[0]:8.3f}"
                  f"  {te[0]:9.3f}  {te[1] * 100:10.1f}%")
            f.write(f"{kind},{v},{tr[1]:.4f},{va[0]:.4f},{te[0]:.4f},"
                    f"{te[1]:.4f}\n")
