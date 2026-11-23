"""渦巻きデータに softmax 回帰を当てはめる（線形モデルの限界を見る）。"""

from pathlib import Path

import torch

from softmax_regression import accuracy, save_grid, save_points, train
from toydata import make_spiral

OUT = Path(__file__).parent / "out"

if __name__ == "__main__":
    # [snippet: spiral]
    X_np, y_np = make_spiral(100, seed=0)  # 3 本の腕が渦を巻くデータ
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    W, b = train(X, y, eta=1.0, n_steps=2000)  # softmax 回帰を学習
    print(f"渦巻きデータの正解率: {accuracy(X, y, W, b):.3f}")
    # [end: spiral]
    save_points(OUT / "spiral_points.csv", X_np, y_np)
    save_grid(OUT / "spiral_grid.csv", W, b, lo=(-1.2, -1.2), hi=(1.2, 1.2))
