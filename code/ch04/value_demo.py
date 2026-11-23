"""自作エンジン Value で、本文の手計算（計算グラフの図）を再現する。"""

from value import Value

# [snippet: demo]
x, y = Value(2.0), Value(5.0)  # データ
w, b = Value(3.0), Value(1.0)  # パラメータ
u = w * x  # 6
y_hat = u + b  # 7
r = y_hat - y  # 2（予測と正解のずれ）
loss = r**2  # 4
loss.backward()
print("loss  =", loss.data)
print("dL/dw =", w.grad)
print("dL/db =", b.grad)
print("dL/dx =", x.grad)
# [end: demo]

print("---")
# [snippet: fanout]
r = Value(2.0)
loss = r * r  # 同じノード r を 2 回使う
loss.backward()
print("dL/dr =", r.grad)  # 2r = 4 になるはず
# [end: fanout]

print("---")


# [snippet: numeric]
def f(w, b, x=2.0, y=5.0):  # ふつうの数で同じ計算をする関数
    return (w * x + b - y) ** 2


h = 1e-6  # 少しだけ動かす幅
print("数値微分 dL/dw =", (f(3.0 + h, 1.0) - f(3.0 - h, 1.0)) / (2 * h))
print("数値微分 dL/db =", (f(3.0, 1.0 + h) - f(3.0, 1.0 - h)) / (2 * h))
# [end: numeric]

print("---")
# [snippet: step]
eta = 0.05  # 学習率
w.data -= eta * w.grad  # 勾配の逆向きに少し動かす
b.data -= eta * b.grad
print("更新後 w =", w.data, " b =", b.data)
print("更新後の損失 =", ((w * x + b - y) ** 2).data)
# [end: step]
