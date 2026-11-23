"""自作エンジン Value だけで MLP を作り、小さな渦巻きデータを学習する。"""

import random
import time

from data import make_spiral
from value import Value


# [snippet: neuron]
class Neuron:
    """ニューロン 1 個: 重み付き和にバイアスを足し、活性化関数を通す。"""

    def __init__(self, n_in, rng, use_relu=True):
        s = n_in**-0.5  # 入力が多いほど重みを小さく始める
        self.w = [Value(rng.uniform(-s, s)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.use_relu = use_relu

    def __call__(self, x):  # x: 長さ n_in のリスト
        u = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return u.relu() if self.use_relu else u
    # [end: neuron]

    def parameters(self):
        return self.w + [self.b]


# [snippet: layer-mlp]
class Layer:
    """層: 同じ入力を受け取るニューロンを n_out 個並べたもの。"""

    def __init__(self, n_in, n_out, rng, use_relu=True):
        self.neurons = [Neuron(n_in, rng, use_relu) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """多層パーセプトロン。sizes=[2, 8, 8, 3] なら 2 → 8 → 8 → 3。"""

    def __init__(self, sizes, seed=0):
        rng = random.Random(seed)
        last = len(sizes) - 2  # 最後の層の番号（最後だけ ReLU なし）
        self.layers = [Layer(sizes[i], sizes[i + 1], rng, use_relu=i < last)
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # 前の層の出力が、次の層の入力になる
        return x  # ロジット（クラスごとのスコア）

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# [end: layer-mlp]


# [snippet: loss]
def cross_entropy(z, y):
    """ロジット z（Value のリスト）と正解 y から交差エントロピーを作る。"""
    log_sum = sum(zk.exp() for zk in z).log()  # log Σ_k exp(z_k)
    return log_sum - z[y]  # = -log softmax(z)_y


def mean_loss(model, X, Y):
    losses = [cross_entropy(model(x), y) for x, y in zip(X, Y)]
    return sum(losses) / len(losses)  # 全データの平均
# [end: loss]


def count_nodes(root):
    """root から入力側へたどって、計算グラフのノードを数える。"""
    seen, stack = set(), [root]
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(v._prev)
    return len(seen)


def accuracy(model, X, Y):
    hit = 0
    for x, y in zip(X, Y):
        z = [zk.data for zk in model(x)]
        hit += z.index(max(z)) == y
    return hit / len(Y)


if __name__ == "__main__":
    X, Y = make_spiral(n_per_class=30, seed=0)  # 90 点
    model = MLP([2, 8, 8, 3], seed=0)
    params = model.parameters()
    print("パラメータ数:", len(params))
    print("計算グラフのノード数:", count_nodes(mean_loss(model, X, Y)))

    log = open("ch04/out/mlp_scratch_log.csv", "w")
    log.write("step,loss\n")
    t0 = time.time()
    # [snippet: train]
    eta = 1.0  # 学習率
    for step in range(301):
        loss = mean_loss(model, X, Y)  # 順伝播（計算グラフができる）
        for p in params:
            p.grad = 0.0  # 前のステップの勾配を消す
        loss.backward()  # 逆伝播（全パラメータの勾配が求まる）
        if step % 50 == 0:
            acc = accuracy(model, X, Y)
            print(f"step {step:3d}  損失 {loss.data:.4f}  正解率 {acc:.3f}")
        for p in params:
            p.data -= eta * p.grad  # 勾配降下法で更新
        # [end: train]
        log.write(f"{step},{loss.data:.10f}\n")
    sec = time.time() - t0
    log.close()
    with open("ch04/out/mlp_scratch_time.txt", "w") as f:
        f.write(f"{sec / 301 * 1000:.1f}\n")
    print(f"1 ステップあたり {sec / 301 * 1000:.0f} ミリ秒")
