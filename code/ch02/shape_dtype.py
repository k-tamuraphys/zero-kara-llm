"""テンソルの形（shape）と型（dtype）、作り方。"""

import torch

# [snippet: dims]
s = torch.tensor(3.14)  # 数 1 個（0 階）
v = torch.tensor([1.0, 2.0, 3.0])  # ベクトル（1 階）
M = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # 行列（2 階）
T = torch.zeros(2, 3, 4)  # 3 階のテンソル（中身はすべて 0）
print(M.shape)  # 形
for name, t in [("s", s), ("v", v), ("M", M), ("T", T)]:
    print(name, tuple(t.shape), "階数", t.ndim, "要素数", t.numel())
# [end: dims]

print("---")
# [snippet: dtype]
i = torch.tensor([1, 2, 3])  # 整数だけで作ると int64
f = torch.tensor([1.0, 2, 3])  # 小数が 1 つでもあると float32
b = f > 1.5  # 比較の結果は bool
print(i.dtype, f.dtype, b.dtype)
print(b)
print(i.float().dtype, f.long().dtype)  # 型を変換する
print(torch.tensor(0.1).item())  # float32 は 0.1 を正確には持てない
# [end: dtype]

print("---")
# [snippet: create]
print(torch.zeros(2, 3))  # すべて 0。形を引数で指定する
print(torch.ones(2, 3))  # すべて 1
print(torch.arange(6))  # 0 から 5 までの整数
print(torch.linspace(0, 1, 5))  # 0 から 1 までを 5 点で等分
print(torch.eye(3))  # 単位行列
# [end: create]
