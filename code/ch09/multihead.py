"""マルチヘッド自己注意を自作し、llmbook と PyTorch の実装と一致させる。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from capture import Capture
from llmbook.model import CausalSelfAttention, GPTConfig

torch.manual_seed(0)
torch.set_grad_enabled(False)  # この章では学習しないので勾配は不要
cap = Capture()  # print の出力を out/ に保存する（capture.py）


# [snippet: mha-class]
class MultiHeadAttention(nn.Module):
    """因果マスク付きマルチヘッド自己注意（説明用に形を追いやすく書く）。"""

    def __init__(self, d: int, n_head: int, block_size: int):
        super().__init__()
        assert d % n_head == 0  # d をヘッド数で割り切れること
        self.H, self.d_head = n_head, d // n_head
        self.W_q = nn.Linear(d, d, bias=False)  # H 個分の W_q をまとめた行列
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)  # 出力の射影
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)))

    def split(self, z):  # (B, T, d) -> (B, H, T, d_head)
        B, T, _ = z.shape
        return z.view(B, T, self.H, self.d_head).transpose(1, 2)

    def forward(self, x):  # x: (B, T, d)
        B, T, d = x.shape
        q, k, v = self.split(self.W_q(x)), self.split(self.W_k(x)), \
            self.split(self.W_v(x))
        S = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B,H,T,T)
        S = S.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        A = F.softmax(S, dim=-1)  # (B, H, T, T)
        y = A @ v  # (B, H, T, d_head)
        y = y.transpose(1, 2).reshape(B, T, d)  # ヘッドを横に並べ直す
        return self.W_o(y), A  # (B, T, d) と注意の重み
# [end: mha-class]


# ---- 形を追う ----
B, T, d, H = 2, 5, 12, 3
mha = MultiHeadAttention(d, H, block_size=8)
x = torch.randn(B, T, d)
# [snippet: shapes]
q = mha.split(mha.W_q(x))
print("入力 x:          ", tuple(x.shape))
print("ヘッドに分けた q:", tuple(q.shape))
y, A = mha(x)
print("注意の重み A:    ", tuple(A.shape))
print("出力 y:          ", tuple(y.shape))
print("A の各行の和:    ", A.sum(dim=-1)[0, 0])
# [end: shapes]
cap.save("shapes.txt")

# ---- 未来を変えても過去の出力は変わらない ----
# [snippet: causal-check]
x2 = x.clone()
x2[:, 3:] = torch.randn(B, T - 3, d)  # 位置 3 以降（未来）だけを変える
y2, _ = mha(x2)
print("位置 0..2 の出力の差:", f"{(y - y2)[:, :3].abs().max().item():.1e}")
print("位置 3..4 の出力の差:", f"{(y - y2)[:, 3:].abs().max().item():.1e}")
# [end: causal-check]
cap.save("causal_check.txt")

# ---- llmbook の CausalSelfAttention と重みをそろえて比べる ----
# [snippet: compare]
cfg = GPTConfig(block_size=8, n_head=H, n_embd=d)
ref = CausalSelfAttention(cfg)
# qkv は W_q, W_k, W_v の重みを縦に積んだ (3d, d) の行列
ref.qkv.weight.copy_(torch.cat(
    [mha.W_q.weight, mha.W_k.weight, mha.W_v.weight], dim=0))
ref.proj.weight.copy_(mha.W_o.weight)
print(f"llmbook との差: {(ref(x) - y).abs().max().item():.1e}")

# PyTorch の組み込み関数（マスクとスケールを内部で行う）
q, k, v = (mha.split(W(x)) for W in (mha.W_q, mha.W_k, mha.W_v))
y_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
y_sdpa = mha.W_o(y_sdpa.transpose(1, 2).reshape(B, T, d))
print(f"F.scaled_dot_product_attention との差: "
      f"{(y_sdpa - y).abs().max().item():.1e}")
# [end: compare]
cap.save("compare.txt")
