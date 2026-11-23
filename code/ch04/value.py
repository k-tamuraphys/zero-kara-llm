"""自作の自動微分エンジン（スカラー版）。

micrograd (Karpathy, 2020) と同じ設計。数を Value で包んで計算すると、
計算グラフが自動的に記録され、backward() ですべての勾配が求まる。
"""

import math


# [snippet: value-init]
class Value:
    """計算グラフのノード 1 個。スカラー値と、その勾配を持つ。"""

    def __init__(self, data, _prev=(), _op=""):
        self.data = data  # 順伝播で求めた値
        self.grad = 0.0  # 損失をこの値で微分したもの（逆伝播で埋まる）
        self._prev = _prev  # このノードを作るのに使った入力ノード
        self._op = _op  # このノードを作った演算の名前（表示用）
        self._backward = lambda: None  # 勾配を入力ノードに配る関数
    # [end: value-init]

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # [snippet: value-add-mul]
    def __add__(self, other):  # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad  # 局所勾配 1 × 上流の勾配
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):  # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad  # 局所勾配は「相手の値」
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    # [end: value-add-mul]

    # [snippet: value-pow-relu]
    def __pow__(self, k):  # self ** k（k はふつうの数）
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out
    # [end: value-pow-relu]

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    # [snippet: value-exp-log]
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad  # exp の微分は exp 自身

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out
    # [end: value-exp-log]

    # [snippet: value-sugar]
    def __neg__(self):  # -self
        return self * -1.0

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other**-1
    # [end: value-sugar]

    def __radd__(self, other):  # other + self（左がふつうの数のとき）
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __rtruediv__(self, other):  # other / self
        return self**-1 * other

    # [snippet: value-backward]
    def backward(self):
        """このノードを出発点に、グラフ全体の勾配を求める。"""
        order, visited = [], set()

        def visit(v):  # 入力側のノードを先に並べる（トポロジカル順）
            if v not in visited:
                visited.add(v)
                for u in v._prev:
                    visit(u)
                order.append(v)

        visit(self)
        self.grad = 1.0  # 出発点: 損失を損失自身で微分すると 1
        for v in reversed(order):  # 出力側から入力側へ
            v._backward()
    # [end: value-backward]
