"""第 11 章: 小さな混合エキスパート（MoE）層。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modern_gpt import SwiGLU


# [snippet: moe]
class MoE(nn.Module):
    def __init__(self, d, hidden, n_expert, top_k):
        super().__init__()
        self.n_expert, self.top_k = n_expert, top_k
        self.router = nn.Linear(d, n_expert, bias=False)      # ルータ
        self.experts = nn.ModuleList(SwiGLU(d, hidden)
                                     for _ in range(n_expert))
        self.aux_loss = torch.tensor(0.0)
        self.usage = torch.zeros(n_expert)                    # 選ばれた回数

    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(-1, C)                                  # (N, C)
        probs = F.softmax(self.router(x), dim=-1)             # (N, E)
        w, idx = probs.topk(self.top_k, dim=-1)               # (N, k)
        w = w / w.sum(dim=-1, keepdim=True)                   # k 個で和を 1
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)     # e を選んだ行
            if tok.numel() > 0:
                y[tok] += w[tok, slot, None] * expert(x[tok])
        # 負荷分散損失（Switch Transformer）: E · Σ_e f_e P_e
        f = F.one_hot(idx[:, 0], self.n_expert).float().mean(0)
        P = probs.mean(0)
        self.aux_loss = self.n_expert * (f * P).sum()
        self.usage = F.one_hot(idx, self.n_expert).float().sum((0, 1))
        return y.reshape(B, T, C)
# [end: moe]
