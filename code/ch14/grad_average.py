"""勾配の平均 = 大きなバッチの勾配 であることを 1 プロセスで確かめる。"""

import torch

from ch14.common import fixed_batch, load_bin, tiny_gpt

torch.set_num_threads(2)
data = load_bin("train")
model = tiny_gpt()
x, y = fixed_batch(data, step=0, B=8)        # 全体のバッチ (8, 64)


def grad_of(xb, yb) -> torch.Tensor:
    """バッチ (xb, yb) の平均損失の勾配を 1 本のベクトルにして返す。"""
    model.zero_grad()
    _, loss = model(xb, yb)
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


# [snippet: compare]
g_big = grad_of(x, y)                        # 8 件まとめて 1 回
K = 4                                        # 「GPU」の数
xs, ys = x.chunk(K), y.chunk(K)              # 2 件ずつ 4 つに分ける
g_parts = [grad_of(xs[k], ys[k]) for k in range(K)]
g_avg = torch.stack(g_parts).mean(dim=0)     # 4 つの勾配の平均
g_sum = torch.stack(g_parts).sum(dim=0)      # （誤り）平均せず和をとる
# [end: compare]

print(f"パラメータ数            : {g_big.numel():,}")
print(f"大バッチの勾配のノルム  : {g_big.norm():.6f}")
for k in range(K):
    d = (g_parts[k] - g_big).norm() / g_big.norm()
    print(f"  部分 {k} だけの勾配   : ノルム {g_parts[k].norm():.6f}"
          f"  大バッチとの相対差 {d:.3f}")
print(f"平均した勾配のノルム    : {g_avg.norm():.6f}")
print(f"平均 vs 大バッチ 最大差 : {(g_avg - g_big).abs().max():.2e}")
print(f"和   vs 大バッチ ノルム比: {g_sum.norm() / g_big.norm():.3f}")
