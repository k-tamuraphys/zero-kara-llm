"""softmax 回帰で 2 次元の点を 3 クラスに分類する。"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from toydata import make_blobs

OUT = Path(__file__).parent / "out"


# [snippet: model]
def logits_of(X, W, b):
    """ロジット z = W x + b を全データについて計算する。"""
    return X @ W.T + b  # X: (N, 2), W: (K, 2), b: (K,) → (N, K)


def cross_entropy(z, y):
    """交差エントロピー損失: 正解クラスの確率の対数にマイナスを付けて平均。"""
    log_p = z - torch.logsumexp(z, dim=1, keepdim=True)  # log softmax (N, K)
    return -log_p[torch.arange(len(y)), y].mean()  # 各行から正解の列を取り出す
# [end: model]


# [snippet: manual-grad]
def manual_grad(X, y, W, b):
    """手で導いた勾配: dL/dz = p - y を使う。"""
    N, K = len(y), W.shape[0]
    p = torch.softmax(logits_of(X, W, b), dim=1)  # 予測確率 (N, K)
    Y = F.one_hot(y, K).float()  # 正解の one-hot 表現 (N, K)
    dz = (p - Y) / N  # 損失は平均なので N で割る
    return dz.T @ X, dz.sum(dim=0)  # dL/dW: (K, 2),  dL/db: (K,)
# [end: manual-grad]


# [snippet: train]
def train(X, y, K=3, eta=0.5, n_steps=300, log=None):
    """softmax 回帰を勾配降下法で学習し、(W, b) を返す。"""
    W = torch.zeros(K, X.shape[1], requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    for step in range(n_steps + 1):
        loss = cross_entropy(logits_of(X, W, b), y)
        if log is not None:
            log(step, W, b, loss.item())
        loss.backward()
        with torch.no_grad():
            W -= eta * W.grad
            b -= eta * b.grad
        W.grad.zero_()
        b.grad.zero_()
    return W.detach(), b.detach()
# [end: train]


# [snippet: accuracy]
def accuracy(X, y, W, b):
    """正解率: ロジットが最大のクラスを予測とし、正解と一致した割合。"""
    pred = logits_of(X, W, b).argmax(dim=1)
    return (pred == y).float().mean().item()
# [end: accuracy]


def save_grid(path, W, b, lo, hi, n=61):
    """決定領域を描くため、格子点ごとの予測クラスを保存する。"""
    g1, g2 = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    with open(path, "w") as f:
        f.write("x1,x2,cls\n")
        for v2 in g2:
            pts = torch.tensor([[v1, v2] for v1 in g1], dtype=torch.float32)
            cls = logits_of(pts, W, b).argmax(dim=1)
            for v1, c in zip(g1, cls):
                f.write(f"{v1:.3f},{v2:.3f},{c.item()}\n")


def save_points(path, X, y):
    with open(path, "w") as f:
        f.write("x1,x2,cls\n")
        for (v1, v2), c in zip(X.tolist(), y.tolist()):
            f.write(f"{v1:.3f},{v2:.3f},{c}\n")


if __name__ == "__main__":
    X_np, y_np = make_blobs(100, seed=0)
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    print("X の形:", tuple(X.shape), " y の形:", tuple(y.shape))
    print("先頭のデータ:", [round(v, 2) for v in X[0].tolist()],
          " ラベル:", y[0].item())

    # 勾配の照合: 手で導いた式と自動微分
    torch.manual_seed(0)
    W0 = torch.randn(3, 2, requires_grad=True)
    b0 = torch.randn(3, requires_grad=True)
    loss = cross_entropy(logits_of(X, W0, b0), y)
    loss.backward()
    gW, gb = manual_grad(X, y, W0.detach(), b0.detach())
    print(f"勾配の差の最大値: W {(gW - W0.grad).abs().max().item():.1e}"
          f"  b {(gb - b0.grad).abs().max().item():.1e}")
    print(f"自作の損失 {loss.item():.5f}  F.cross_entropy "
          f"{F.cross_entropy(logits_of(X, W0, b0), y).item():.5f}")

    curve = []

    def log(step, W, b, loss):
        acc = accuracy(X, y, W.detach(), b.detach())
        curve.append((step, loss, acc))
        if step in (0, 1, 10, 100, 300):
            print(f"step {step:3d}  損失 = {loss:.4f}  正解率 = {acc:.3f}")

    W, b = train(X, y, log=log)
    print("W =", [[round(v, 2) for v in row] for row in W.tolist()])
    print("b =", [round(v, 2) for v in b.tolist()])

    X_test_np, y_test_np = make_blobs(100, seed=1)  # 学習に使っていない点
    X_test, y_test = torch.from_numpy(X_test_np), torch.from_numpy(y_test_np)
    print(f"訓練データの正解率 {accuracy(X, y, W, b):.3f}  "
          f"新しいデータの正解率 {accuracy(X_test, y_test, W, b):.3f}")
    # [snippet: predict]
    x_new = torch.tensor([[-2.0, -1.0], [0.0, 0.0]])  # 新しい点を 2 つ
    p_new = torch.softmax(logits_of(x_new, W, b), dim=1)  # (2, 3)
    for x_i, p_i in zip(x_new.tolist(), p_new.tolist()):
        print("点", x_i, "の予測確率:", [round(v, 3) for v in p_i])
    # [end: predict]

    with open(OUT / "softmax_curve.csv", "w") as f:
        f.write("step,loss,acc\n")
        for step, loss_v, acc in curve:
            f.write(f"{step},{loss_v:.5f},{acc:.4f}\n")
    save_points(OUT / "blobs_points.csv", X_np, y_np)
    save_grid(OUT / "blobs_grid.csv", W, b, lo=(-5.0, -4.0), hi=(5.0, 5.0))
