"""2 次元の分類用の人工データ。"""

import numpy as np


# [snippet: blobs]
def make_blobs(n_per_class: int = 100, seed: int = 0):
    """3 つの塊からなるデータ。X: (3n, 2), y: (3n,) でラベルは 0, 1, 2。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])  # 塊の中心
    X = np.concatenate(
        [c + rng.normal(size=(n_per_class, 2)) for c in centers])  # 中心+ずれ
    y = np.repeat(np.arange(3), n_per_class)  # 0,0,...,1,1,...,2,2,...
    return X.astype(np.float32), y
# [end: blobs]


def make_spiral(n_per_class: int = 100, seed: int = 0):
    """3 本の腕が渦を巻くデータ（直線では分けられない例）。"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(3):
        r = np.linspace(0.05, 1.0, n_per_class)  # 中心からの距離
        t = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class)  # 角度
        t = t + rng.normal(size=n_per_class) * 0.2
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        y.append(np.full(n_per_class, k))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)
