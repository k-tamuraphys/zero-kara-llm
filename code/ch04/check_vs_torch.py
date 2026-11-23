"""自作エンジン Value の勾配が PyTorch の autograd と一致するか確かめる。"""

import torch

from value import Value


# [snippet: expr]
def expr(a, b, c):
    """足し算・掛け算・割り算・べき乗・tanh・exp・log・ReLU を全部使う式。"""
    d = a * b + c**3
    e = (d / a - 1.0).tanh()
    f = (e * 2.0).exp() + (a * a + 1.0).log()
    g = (f - b).relu() + c * d  # d は 2 か所で使われる（分岐）
    return g
# [end: expr]


vals = [1.5, -2.0, 0.7]  # a, b, c の値

# 自作エンジン
a, b, c = [Value(v) for v in vals]
g = expr(a, b, c)
g.backward()

# PyTorch（Value と同じ倍精度にそろえる）
ta, tb, tc = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
              for v in vals]
tg = expr(ta, tb, tc)  # 同じ関数がそのまま使える
tg.backward()

print(f"g      自作 {g.data:.12f}  PyTorch {tg.item():.12f}")
for name, v, t in [("dg/da", a, ta), ("dg/db", b, tb), ("dg/dc", c, tc)]:
    print(f"{name}  自作 {v.grad:.12f}  PyTorch {t.grad.item():.12f}")
