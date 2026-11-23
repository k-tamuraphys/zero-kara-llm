"""第 5 章: 深い MLP を、初期化・正規化・残差接続の有無を変えて学習させる。

隠れ層 10・20・50 枚、幅 32、SGD（学習率 0.1）、1,000 ステップ。
    cd code && python ch05/deep_train.py > ch05/out/deep_train.txt
乱数の種を変えるには python ch05/deep_train.py 1 のように書く
（出力ファイル名は自分で変える。曲線の CSV は種 0 のときだけ書く）。
結果は out/deep_train_seed<種>.csv にも書き、deep_summary.py が集計する。
"""
import sys
import time

import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, evaluate, load_spiral

torch.set_num_threads(1)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS, LR = 1000, 0.1
X, Y, Xt, Yt = load_spiral()

CONFIGS = [
    ("default", "PyTorch の標準の初期値", dict(init="default")),
    ("he", "He の初期化", dict(init="he")),
    ("he_ln", "He＋レイヤー正規化", dict(init="he", norm=True)),
    ("he_res", "He＋残差接続", dict(init="he", residual=True)),
    ("he_res_ln", "He＋レイヤー正規化＋残差接続",
     dict(init="he", norm=True, residual=True)),
]


def train(depth, **kw):
    torch.manual_seed(SEED)
    model = DeepMLP(depth, **kw)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    curve = []
    for step in range(STEPS):
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        curve.append(loss.item())
        if not torch.isfinite(loss):         # 損失が nan や inf になったら中止
            return curve, None, None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return curve, evaluate(model, X, Y), evaluate(model, Xt, Yt)


def pad(s, width):
    return s + " " * (width - sum(2 if ord(c) > 255 else 1 for c in s))


t0 = time.time()
curves = {}
rows = []
print(f"幅 32、SGD（学習率 {LR}）、{STEPS} ステップ、乱数の種 {SEED}")
print("学習後の訓練損失（訓練の正解率 / テストの正解率）")
for depth in (10, 20, 50):
    print(f"-- 隠れ層 {depth} 枚")
    for key, name, kw in CONFIGS:
        curve, train_res, test_res = train(depth, **kw)
        curves[(depth, key)] = curve
        rows.append((depth, key) + (train_res or ("nan", "nan"))
                    + ((test_res[1],) if test_res else ("nan",)))
        if train_res is None:
            result = f"発散（ステップ {len(curve) - 1} で損失が nan か inf）"
        else:
            result = (f"{train_res[0]:.3f}（{train_res[1] * 100:.0f}% /"
                      f" {test_res[1] * 100:.0f}%）")
        print(pad(name, 30) + f"最初の損失 {curve[0]:9.3g}  → {result}")

with open(f"ch05/out/deep_train_seed{SEED}.csv", "w") as f:
    f.write("depth,config,train_loss,train_acc,test_acc\n")
    for row in rows:
        f.write(",".join(str(v) for v in row) + "\n")
if SEED == 0:
    with open("ch05/out/deep_train_curves.csv", "w") as f:
        keys = ["he", "he_ln", "he_res_ln"]
        f.write("step," + ",".join(keys) + "\n")
        for i in range(0, STEPS, 5):
            f.write(f"{i}," + ",".join(
                f"{curves[(50, k)][i]:.5f}" if i < len(curves[(50, k)])
                and curves[(50, k)][i] == curves[(50, k)][i] else "nan"
                for k in keys) + "\n")
print(f"実行時間 {time.time() - t0:.0f} 秒", file=sys.stderr)
