"""自作エンジンの MLP と、同じ初期値の PyTorch 版を比べる。

(1) 最初のステップの勾配が全パラメータで一致するか。
(2) 同じ学習率で学習させたとき、損失の推移が一致するか。
(3) 1 ステップにかかる時間はどれだけ違うか。
先に mlp_scratch.py を実行しておくこと（損失の記録を読む）。
"""

import time

import torch
import torch.nn.functional as F

from data import make_spiral
from mlp_scratch import MLP, mean_loss

torch.set_num_threads(1)
X, Y = make_spiral(n_per_class=30, seed=0)
model = MLP([2, 8, 8, 3], seed=0)  # mlp_scratch.py と同じ初期値

# --- 自作エンジンで勾配を求める
loss_s = mean_loss(model, X, Y)
loss_s.backward()

# [snippet: copy]
# 自作 MLP の重みを、同じ値の PyTorch テンソルに写す（倍精度）
f64 = torch.float64
Ws = [torch.tensor([[w.data for w in n.w] for n in layer.neurons],
                   dtype=f64, requires_grad=True) for layer in model.layers]
bs = [torch.tensor([n.b.data for n in layer.neurons],
                   dtype=f64, requires_grad=True) for layer in model.layers]
Xt, Yt = torch.tensor(X, dtype=f64), torch.tensor(Y)


def forward(x):  # x: (N, 2)
    h = torch.relu(x @ Ws[0].T + bs[0])  # (N, 8)
    h = torch.relu(h @ Ws[1].T + bs[1])  # (N, 8)
    return h @ Ws[2].T + bs[2]  # (N, 3) ロジット


loss_t = F.cross_entropy(forward(Xt), Yt)
loss_t.backward()
# [end: copy]

# [snippet: compare]
print(f"損失    自作 {loss_s.data:.12f}  PyTorch {loss_t.item():.12f}")
max_diff = 0.0
for layer, W, b in zip(model.layers, Ws, bs):
    for i, n in enumerate(layer.neurons):
        for j, w in enumerate(n.w):
            max_diff = max(max_diff, abs(w.grad - W.grad[i, j].item()))
        max_diff = max(max_diff, abs(n.b.grad - b.grad[i].item()))
n_params = sum(W.numel() + b.numel() for W, b in zip(Ws, bs))
print(f"勾配    {n_params} 個のパラメータでの差の最大値 {max_diff:.2e}")
# [end: compare]

# --- 同じ学習率で PyTorch 版を学習させ、損失を記録する
scratch = [float(line.split(",")[1])
           for line in open("ch04/out/mlp_scratch_log.csv").readlines()[1:]]
eta, torch_log = 1.0, []
t0 = time.time()
for step in range(301):
    loss = F.cross_entropy(forward(Xt), Yt)
    for p in Ws + bs:
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in Ws + bs:
            p -= eta * p.grad
    torch_log.append(loss.item())
ms_torch = (time.time() - t0) / 301 * 1000
with open("ch04/out/scratch_vs_torch.csv", "w") as f:
    f.write("step,scratch,torch\n")
    for step, (a, b) in enumerate(zip(scratch, torch_log)):
        f.write(f"{step},{a:.10f},{b:.10f}\n")
diff = max(abs(a - b) for a, b in zip(scratch, torch_log))
print(f"学習    301 ステップの損失の差の最大値 {diff:.2e}")
for step in (0, 100, 200, 300):
    print(f"        step {step:3d}  自作 {scratch[step]:.6f}"
          f"  PyTorch {torch_log[step]:.6f}")
ms_scratch = float(open("ch04/out/mlp_scratch_time.txt").read())
print(f"時間    自作 {ms_scratch:.0f} ミリ秒/ステップ"
      f"  PyTorch {ms_torch:.2f} ミリ秒/ステップ"
      f"（約 {ms_scratch / ms_torch:.0f} 倍）")
