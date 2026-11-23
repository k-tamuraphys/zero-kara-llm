"""自動微分のさわり: requires_grad と backward()。"""

import torch

# [snippet: basic]
x = torch.tensor([2.0, 1.0], requires_grad=True)  # 「この変数で微分する」印
f = x[0] ** 2 + 3 * x[0] * x[1]  # f(x1, x2) = x1^2 + 3 x1 x2 を普通に書く
print(f)  # 値は 2^2 + 3*2*1 = 10
f.backward()  # f を x の各成分で偏微分する
print(x.grad)  # 結果（勾配）は x.grad に入る
# [end: basic]

print("---")


# [snippet: numeric]
def f_plain(x1, x2):
    return x1 ** 2 + 3 * x1 * x2


h = 1e-4  # 少しだけ動かす幅
d1 = (f_plain(2.0 + h, 1.0) - f_plain(2.0 - h, 1.0)) / (2 * h)
d2 = (f_plain(2.0, 1.0 + h) - f_plain(2.0, 1.0 - h)) / (2 * h)
print(d1, d2)  # 数値微分でも、ほぼ同じ値になる
# [end: numeric]

print("---")
# [snippet: accumulate]
g = (x ** 2).sum()  # 別の関数 g = x1^2 + x2^2。勾配は (2 x1, 2 x2) = (4, 2)
g.backward()
print(x.grad)  # 前の (7, 6) に (4, 2) が足し込まれてしまう
x.grad.zero_()  # 勾配を 0 に戻してから……
g = (x ** 2).sum()
g.backward()
print(x.grad)  # 今度は正しく (4, 2)
# [end: accumulate]

print("---")
# [snippet: step]
with torch.no_grad():  # この中の計算は、微分の対象として記録しない
    x_new = x - 0.1 * x.grad  # 勾配の逆向きに、少しだけ動く
print(x_new)
g_old, g_new = (x ** 2).sum(), (x_new ** 2).sum()
print(f"g: {g_old:.2f} → {g_new:.2f}")  # g の値は小さくなる
# [end: step]

print("---")
# [snippet: matrix]
torch.manual_seed(0)
W = torch.randn(2, 3, requires_grad=True)  # 形 (2, 3) の変数
X = torch.randn(4, 3)  # 形 (4, 3) のデータ（こちらは定数）
loss = ((X @ W.T) ** 2).mean()  # W から決まる、数 1 個の値
loss.backward()
print(W.grad.shape)  # 勾配は W と同じ形になる
# [end: matrix]
