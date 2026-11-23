"""第 4 章で使う玩具データ: XOR と、3 クラスの渦巻き。

外部ライブラリを使わず、Python のリストで返す
（自作エンジンでも PyTorch でも同じデータを使えるようにするため）。
"""

import math
import random

# XOR: 2 つの入力の「どちらか一方だけが 1」のときクラス 1
XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [0, 1, 1, 0]


# [snippet: spiral]
def make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=0):
    """原点から渦を巻いて伸びる n_classes 本の腕。腕ごとにクラスが違う。"""
    rng = random.Random(seed)
    X, y = [], []
    for k in range(n_classes):
        for i in range(n_per_class):
            r = (i + 1) / n_per_class  # 半径: 原点から 1 まで
            t = 2 * math.pi * k / n_classes + 4.0 * r  # 角度: 外ほど回る
            t += rng.gauss(0.0, noise)  # 角度に小さな乱れを加える
            X.append([r * math.sin(t), r * math.cos(t)])
            y.append(k)
    return X, y
# [end: spiral]
