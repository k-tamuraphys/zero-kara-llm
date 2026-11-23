"""PyTorch の autograd で、本文の手計算と同じ勾配を求める。"""

import torch

# [snippet: autograd]
x, y = torch.tensor(2.0), torch.tensor(5.0)  # データ（勾配は不要）
w = torch.tensor(3.0, requires_grad=True)  # パラメータ（勾配がほしい）
b = torch.tensor(1.0, requires_grad=True)
loss = (w * x + b - y) ** 2  # 順伝播。裏で計算グラフが記録される
loss.backward()  # 逆伝播
print("loss  =", loss.item())
print("dL/dw =", w.grad.item())
print("dL/db =", b.grad.item())
print("grad_fn:", loss.grad_fn)  # loss を作った演算の記録
# [end: autograd]

print("---")
# [snippet: accumulate]
loss = (w * x + b - y) ** 2  # もう一度、同じ計算をする
loss.backward()  # 勾配を消さずに逆伝播すると……
print("2 回目の backward の後  dL/dw =", w.grad.item())
w.grad.zero_()  # 勾配を 0 に戻す
b.grad.zero_()
loss = (w * x + b - y) ** 2
loss.backward()
print("0 に戻してからの backward  dL/dw =", w.grad.item())
# [end: accumulate]
