"""第 5 章: 隠れ層 50 枚の MLP で、表 5.1 を補う確認の実験。

(1) 正規化の位置と残差接続を別々に変える。
(2) 「He＋レイヤー正規化」（残差接続なし）で、学習率と手法を変える。
どれも幅 32、1,000 ステップ、全データで学習（deep_train.py と同じ）。
    cd code && python ch05/deep_ablation.py 0     # 乱数の種 0 で学習
    cd code && python ch05/deep_ablation.py summary \
        > ch05/out/deep_ablation.txt
乱数の種 0〜4 の結果を out/deep_ablation_seed<種>.csv に書き、
summary で集計する。
"""
import csv
import sys

import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, evaluate, load_spiral

torch.set_num_threads(1)
DEPTH, STEPS, SEEDS = 50, 1000, range(5)


class VariantMLP(DeepMLP):
    """正規化の位置（pre）と残差接続（skip）を別々に選べる DeepMLP。"""

    def __init__(self, pre, skip, last):
        super().__init__(DEPTH, init="he", norm=True, residual=last)
        self.pre, self.skip = pre, skip

    def forward(self, x):
        h = torch.relu(self.first(x))
        for layer, norm in zip(self.layers, self.norms):
            if self.pre:
                f = torch.relu(layer(norm(h)))       # 正規化を前に置く
            else:
                f = torch.relu(norm(layer(h)))       # 正規化を後ろに置く
            h = h + f if self.skip else f
        return self.out(self.last_norm(h))


def sgd(lr):
    return lambda params: torch.optim.SGD(params, lr=lr)


def adam(lr):
    return lambda params: torch.optim.Adam(params, lr=lr)


# (名前, モデルを作る関数, オプティマイザを作る関数)
CONFIGS = [
    ("正規化を前に置く・残差接続なし",
     lambda: VariantMLP(pre=True, skip=False, last=True), sgd(0.1)),
    ("正規化を後ろに置く・残差接続あり",
     lambda: VariantMLP(pre=False, skip=True, last=False), sgd(0.1)),
    ("正規化を前に置く・残差接続あり（出力前の正規化なし）",
     lambda: VariantMLP(pre=True, skip=True, last=False), sgd(0.1)),
    ("He＋レイヤー正規化、SGD 0.01",
     lambda: DeepMLP(DEPTH, init="he", norm=True), sgd(0.01)),
    ("He＋レイヤー正規化、SGD 0.03",
     lambda: DeepMLP(DEPTH, init="he", norm=True), sgd(0.03)),
    ("He＋レイヤー正規化、Adam 0.001",
     lambda: DeepMLP(DEPTH, init="he", norm=True), adam(0.001)),
    ("He＋レイヤー正規化、Adam 0.01",
     lambda: DeepMLP(DEPTH, init="he", norm=True), adam(0.01)),
]


def train(seed, make_model, make_optimizer):
    X, Y, Xt, Yt = load_spiral()
    torch.manual_seed(seed)
    model = make_model()
    optimizer = make_optimizer(model.parameters())
    for step in range(STEPS):
        loss = F.cross_entropy(model(X), Y)
        if not torch.isfinite(loss):
            return "nan", "nan", "nan"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, X, Y) + (evaluate(model, Xt, Yt)[1],)


def pad(s, width):
    return s + " " * (width - sum(2 if ord(c) > 255 else 1 for c in s))


if sys.argv[1] == "summary":
    rows = {name: [] for name, _, _ in CONFIGS}
    for seed in SEEDS:
        with open(f"ch05/out/deep_ablation_seed{seed}.csv") as f:
            for r in csv.DictReader(f):
                rows[r["config"]].append(r)
    print(f"隠れ層 {DEPTH} 枚、乱数の種 {len(SEEDS)} 通りのうち、"
          "学習できた回数（括弧内は発散した回数）")
    for name, rs in rows.items():
        ok = sum(r["train_acc"] != "nan" and float(r["train_acc"]) >= 0.99
                 for r in rs)
        div = sum(r["train_acc"] == "nan" for r in rs)
        accs = " ".join("  発散" if r["train_acc"] == "nan"
                        else f"{float(r['train_acc']) * 100:5.1f}%"
                        for r in rs)
        print(pad(name, 54) + f"{ok}（{div}）  訓練の正解率: {accs}")
else:
    seed = int(sys.argv[1])
    with open(f"ch05/out/deep_ablation_seed{seed}.csv", "w") as f:
        f.write("config,train_loss,train_acc,test_acc\n")
        for name, make_model, make_optimizer in CONFIGS:
            res = train(seed, make_model, make_optimizer)
            f.write(name + "," + ",".join(str(v) for v in res) + "\n")
            f.flush()
            print(seed, name, res, flush=True)
