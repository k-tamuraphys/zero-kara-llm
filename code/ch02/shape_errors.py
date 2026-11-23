"""よくある形のエラーをわざと起こし、エラー文を読む。"""

import torch


def show(title, f):
    """f() を実行し、エラーが出たらその種類と文面を表示する。"""
    try:
        f()
    except Exception as e:  # noqa: BLE001
        print(f"[{title}]")
        print(f"  {type(e).__name__}: {e}")


# [snippet: errors]
X = torch.zeros(4, 3)  # 4 個のデータ。形 (4, 3)
W = torch.zeros(2, 3)  # 形 (2, 3)
S = torch.zeros(2, 3)  # 点数の表。形 (2, 3)
m = S.mean(dim=1)  # keepdim=True を付け忘れた平均。形 (2,)
show("1. 行列積（転置の付け忘れ）", lambda: X @ W)
show("2. 引き算（keepdim の付け忘れ）", lambda: S - m)
show("3. 形の変更", lambda: torch.arange(12).reshape(5, -1))
show("4. 番号", lambda: X[4])
show("5. 型", lambda: X @ torch.ones(3, 2, dtype=torch.int64))
# [end: errors]

print("---")


# [snippet: affine]
def affine(X, W, b):
    """各行 x に W x + b を適用する。X: (N, D_in) → (N, D_out)"""
    N, D_in = X.shape
    D_out = W.shape[0]
    assert W.shape == (D_out, D_in), f"W の形が {tuple(W.shape)}"
    assert b.shape == (D_out,), f"b の形が {tuple(b.shape)}"
    Y = X @ W.T + b  # (N, D_in) @ (D_in, D_out) + (D_out,)
    assert Y.shape == (N, D_out)
    return Y
# [end: affine]


# [snippet: assert]
X = torch.ones(2, 3)
W = torch.ones(2, 3)
b_bad = torch.tensor([[10.0], [20.0]])  # 形 (2, 1)。本当は (2,) のつもり
print(X @ W.T + b_bad)  # assert がないと、黙って間違った答えが出る
try:
    affine(X, W, b_bad)
except AssertionError as e:
    print("AssertionError:", e)
print(affine(X, W, b_bad.squeeze(1)))  # 形を直せば、正しい答えが出る
# [end: assert]
