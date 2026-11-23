"""注意機構を一歩ずつ組み立てる: 平均 → 重み付き平均 → 縮小付き内積注意。"""

import math

import torch
import torch.nn.functional as F

from capture import Capture

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)
cap = Capture()  # print の出力を out/ に保存する（capture.py）

# ---- 1. 過去の平均を for 文で計算する ----
# [snippet: average-loop]
T, d = 4, 2
X = torch.randn(T, d)  # T 個のトークンのベクトル（1 トークンが 1 行）
Y = torch.zeros(T, d)
for t in range(T):
    Y[t] = X[: t + 1].mean(dim=0)  # 自分と過去（0..t 行目）の平均
# [end: average-loop]
print("Y（for 文）の 2 行目:", Y[1])

# ---- 2. 同じ平均を 1 回の行列積で計算する ----
# [snippet: average-matrix]
A = torch.tril(torch.ones(T, T))  # 下三角が 1、上三角が 0
A = A / A.sum(dim=1, keepdim=True)  # 各行の和が 1 になるよう割る
Y2 = A @ X  # (T, T) @ (T, d) -> (T, d)
# [end: average-matrix]
print("A =")
print(A)
print("for 文と行列積は一致するか:", torch.allclose(Y, Y2))
cap.save("average.txt")

# ---- 3. 手計算の例: 3 トークン、2 次元 ----
# [snippet: tiny-example]
Q = torch.tensor([[1.0, 0.0],   # 1 番目のトークンのクエリ
                  [0.0, 2.0],   # 2 番目
                  [2.0, 0.0]])  # 3 番目
K = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
V = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0]])
d_head = Q.size(1)
S = Q @ K.T / math.sqrt(d_head)  # スコア (3, 3)
mask = torch.tril(torch.ones(3, 3))
S = S.masked_fill(mask == 0, float("-inf"))  # 未来を -∞ に
A = F.softmax(S, dim=-1)  # 各行を確率に（行の和が 1）
Y = A @ V  # 重み付き平均 (3, 2)
# [end: tiny-example]
print("注意の重み A =")
print(A)
print("出力 Y =")
print(Y)
cap.save("tiny_example.txt")

# ---- 4. 自己注意: for 文版と行列版 ----
torch.manual_seed(0)
T, d, d_head = 5, 8, 4
X = torch.randn(T, d)  # 入力（1 トークンが 1 行）
W_q, W_k, W_v = (torch.randn(d, d_head) for _ in range(3))


# [snippet: attention-loop]
def attention_loop(X, W_q, W_k, W_v):
    """位置 t ごと・相手 s ごとに、1 つずつ計算する素朴な版。"""
    T = X.size(0)
    Y = torch.zeros(T, W_v.size(1))
    for t in range(T):
        q = X[t] @ W_q  # 位置 t のクエリ (d_head,)
        scores = []
        for s in range(t + 1):  # 自分と過去だけを見る
            k = X[s] @ W_k  # 位置 s のキー
            scores.append(q @ k / math.sqrt(len(q)))
        a = F.softmax(torch.stack(scores), dim=0)  # (t+1,)
        for s in range(t + 1):
            Y[t] += a[s] * (X[s] @ W_v)  # バリューの重み付き和
    return Y
# [end: attention-loop]


# [snippet: attention-matrix]
def attention_matrix(X, W_q, W_k, W_v):
    """同じ計算を行列積でまとめて行う版。"""
    T = X.size(0)
    Q, K, V = X @ W_q, X @ W_k, X @ W_v  # どれも (T, d_head)
    S = Q @ K.T / math.sqrt(Q.size(1))  # (T, T)
    mask = torch.tril(torch.ones(T, T))
    S = S.masked_fill(mask == 0, float("-inf"))
    A = F.softmax(S, dim=-1)  # (T, T)、各行の和が 1
    return A @ V  # (T, d_head)
# [end: attention-matrix]


Y1 = attention_loop(X, W_q, W_k, W_v)
Y2 = attention_matrix(X, W_q, W_k, W_v)
print("for 文版の出力の最後の行:", Y1[-1])
print("行列版の出力の最後の行:", Y2[-1])
print(f"最大の差: {(Y1 - Y2).abs().max().item():.1e}")
print("一致するか:", torch.allclose(Y1, Y2, atol=1e-5))
cap.save("loop_vs_matrix.txt")
