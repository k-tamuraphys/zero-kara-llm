"""第 5 章: 学習率スケジュール（ウォームアップ＋コサイン減衰）。

(1) 基盤モデルの学習で使った設定のスケジュールを表にする。
(2) 隠れ層 20 枚の MLP を Adam・高めの学習率 0.1 で学習させ、
    スケジュールの有無を比べる（ミニバッチ 32 件、400 ステップ、種 3 個）。
    cd code && python ch05/schedule.py > ch05/out/schedule.txt
"""
import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, evaluate, load_spiral
from llmbook.train import lr_at

torch.set_num_threads(1)
X, Y, Xt, Yt = load_spiral()

print("基盤モデルの設定: 最大 0.001、ウォームアップ 200、全 4000 ステップ")
for step in (0, 99, 199, 200, 1000, 2100, 3000, 3999):
    print(f"ステップ {step:4d}: 学習率 {lr_at(step, 1e-3, 200, 4000):.6f}")
with open("ch05/out/schedule.csv", "w") as f:
    f.write("step,lr\n")
    for step in list(range(0, 4000, 20)) + [3999]:
        f.write(f"{step},{lr_at(step, 1e-3, 200, 4000):.8f}\n")
print("---")

STEPS, MAX_LR, BATCH = 400, 0.1, 32
MODES = {                      # 名前: (ウォームアップ, 最後の学習率の割合)
    "一定": (0, 1.0),
    "コサイン減衰だけ": (0, 0.1),
    "ウォームアップ＋コサイン減衰": (40, 0.1),
}


def train(mode, seed):
    warmup, min_ratio = MODES[mode]
    torch.manual_seed(seed)
    model = DeepMLP(20, init="he", norm=True, residual=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    losses = []
    # [snippet: set-lr]
    for step in range(STEPS):
        lr = lr_at(step, MAX_LR, warmup, STEPS, min_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr                 # このステップの学習率を設定
        idx = torch.randint(0, len(X), (BATCH,))     # ミニバッチを選ぶ
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # [end: set-lr]
        losses.append(loss.item())
    return losses, evaluate(model, X, Y), evaluate(model, Xt, Yt)


def pad(s, width):
    return s + " " * (width - sum(2 if ord(c) > 255 else 1 for c in s))


print(f"隠れ層 20 枚、Adam、最大学習率 {MAX_LR}、{STEPS} ステップ")
print("（乱数の種 0, 1, 2 の順。最後の行は 3 回の平均）")
curves = {}
for mode in MODES:
    rows = []
    for seed in range(3):
        losses, tr, te = train(mode, seed)
        if seed == 0:
            curves[mode] = losses
        rows.append((max(losses[:50]), tr[0], te[1] * 100))
    print(mode)
    for r in rows + [tuple(sum(c) / 3 for c in zip(*rows))]:
        print(f"  最初の 50 ステップの損失の最大 {r[0]:5.2f}"
              f"  最後の訓練損失 {r[1]:.4f}  テスト正解率 {r[2]:5.1f}%")
with open("ch05/out/schedule_curves.csv", "w") as f:
    f.write("step,const,cos,warmcos\n")
    for i in range(STEPS):
        f.write(f"{i}," + ",".join(f"{curves[m][i]:.5f}" for m in MODES)
                + "\n")
