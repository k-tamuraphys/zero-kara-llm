"""自作エンジンの学習（mlp_scratch.py）で、学習率だけを 4 通りに変える。

自作エンジンは遅いので、scratch_vs_torch.py と同じやり方で、
同じ初期値の重みを PyTorch の倍精度テンソルに写して学習させる
（自作エンジンと同じ計算で、損失の差は丸め誤差の大きさ）。
固定: データ 90 点、MLP([2, 8, 8, 3])、初期値（種 0）、300 回の更新。
出力: ch04/out/sweep_lr_scratch.csv（ステップごとの損失）
"""

import torch
import torch.nn.functional as F

from data import make_spiral
from mlp_scratch import MLP

torch.set_num_threads(2)
LRS = (0.1, 0.3, 1.0, 2.0)
X, Y = make_spiral(n_per_class=30, seed=0)
model = MLP([2, 8, 8, 3], seed=0)  # mlp_scratch.py と同じ初期値
f64 = torch.float64
W0 = [torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=f64)
      for layer in model.layers]
b0 = [torch.tensor([n.b.data for n in layer.neurons], dtype=f64)
      for layer in model.layers]
Xt, Yt = torch.tensor(X, dtype=f64), torch.tensor(Y)

logs = {}
for eta in LRS:
    Ws = [W.clone().requires_grad_() for W in W0]  # 毎回、同じ初期値から
    bs = [b.clone().requires_grad_() for b in b0]
    log = []
    for step in range(301):
        h = torch.relu(Xt @ Ws[0].T + bs[0])  # (90, 8)
        h = torch.relu(h @ Ws[1].T + bs[1])  # (90, 8)
        loss = F.cross_entropy(h @ Ws[2].T + bs[2], Yt)
        log.append(loss.item())
        for p in Ws + bs:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in Ws + bs:
                p -= eta * p.grad
    logs[eta] = log
    # 「跳ね上がり」: 前のステップより損失が 0.1 以上増えた回数
    jumps = sum(1 for a, b in zip(log, log[1:]) if b - a > 0.1)
    print(f"学習率 {eta:3.1f}: 最初 {log[0]:.4f}  最後 {log[-1]:.4f}  "
          f"最大 {max(log):.4f}  0.1 以上増えた回数 {jumps}")

# 学習率 1.0 は mlp_scratch.py と同じ設定。その記録と比べて確かめる
scratch = [float(line.split(",")[1]) for line in
           open("ch04/out/mlp_scratch_log.csv").readlines()[1:]]
diff = max(abs(a - b) for a, b in zip(scratch, logs[1.0]))
print(f"学習率 1.0 と mlp_scratch_log.csv の差の最大値 {diff:.1e}")

with open("ch04/out/sweep_lr_scratch.csv", "w") as f:
    f.write("step," + ",".join(f"lr{eta}" for eta in LRS) + "\n")
    for i in range(301):
        f.write(f"{i}," + ",".join(f"{logs[e][i]:.5f}" for e in LRS) + "\n")
