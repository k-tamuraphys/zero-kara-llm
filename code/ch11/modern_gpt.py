"""第 11 章: 現代版ミニGPT（RMSNorm + RoPE + SwiGLU + GQA + KV キャッシュ）。

llmbook.model.GPT（GPT-2 型）との違いだけを書き換えた実装。
MoE の FFN は ch11/moe.py にあり、n_expert > 0 のときに使う。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModernConfig:
    vocab_size: int = 4096
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4         # クエリのヘッド数 H
    n_kv_head: int = 2      # キー・バリューのヘッド数 H_kv（GQA）
    n_embd: int = 192
    ffn_hidden: int = 512   # SwiGLU の中間次元（約 8/3 d）
    rope_theta: float = 10000.0
    tie_weights: bool = False
    n_expert: int = 0       # 0 なら密な SwiGLU。>0 なら MoE（moe.py）
    top_k: int = 2


# [snippet: rmsnorm]
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))  # γ（バイアス β はない）

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
# [end: rmsnorm]


# [snippet: rope]
def rope_tables(head_dim: int, max_len: int, theta: float = 10000.0):
    """各位置 m・各ペア i の角度 m*ω_i の cos と sin を前計算する。"""
    i = torch.arange(head_dim // 2)
    omega = theta ** (-2 * i / head_dim)                # (dh/2,)
    angle = torch.arange(max_len)[:, None] * omega      # (T, dh/2)
    return angle.cos(), angle.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, dh)。隣り合う 2 成分を 1 組として角度だけ回す。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]                 # 各組の 1・2 成分
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).flatten(-2)    # 元の並びに戻す
# [end: rope]


# [snippet: swiglu]
class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
# [end: swiglu]


# [snippet: gqa]
class GQAttention(nn.Module):
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        self.h, self.hkv = cfg.n_head, cfg.n_kv_head
        self.dh = cfg.n_embd // cfg.n_head
        self.q = nn.Linear(cfg.n_embd, self.h * self.dh, bias=False)
        self.kv = nn.Linear(cfg.n_embd, 2 * self.hkv * self.dh, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.hkv, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)     # (B, Hkv, T, dh)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:                           # KV キャッシュ
            k, v = cache.update(k, v)                   # 過去分とつなぐ
        rep = self.h // self.hkv                        # 1 組を何ヘッドで共有
        k = k.repeat_interleave(rep, dim=1)             # (B, H, S, dh)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.proj(y.transpose(1, 2).reshape(B, T, C))
# [end: gqa]


# [snippet: kvcache]
class KVCache:
    """1 層ぶんのキーとバリューを、生成の間ためておく箱。"""
    def __init__(self):
        self.k = self.v = None

    def update(self, k_new, v_new):                     # (B, Hkv, t, dh)
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)  # 時刻の軸に追加
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v
# [end: kvcache]


# [snippet: block]
class ModernBlock(nn.Module):
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        if cfg.n_expert > 0:
            from moe import MoE
            self.ffn = MoE(cfg.n_embd, cfg.ffn_hidden, cfg.n_expert, cfg.top_k)
        else:
            self.ffn = SwiGLU(cfg.n_embd, cfg.ffn_hidden)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)   # Pre-Norm
        x = x + self.ffn(self.norm2(x))
        return x
# [end: block]


class ModernGPT(nn.Module):
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        self.cfg = self.config = cfg   # train_lm は .config を読む
        # 位置埋め込みの表はない
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            ModernBlock(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_tables(cfg.n_embd // cfg.n_head, 4 * cfg.block_size,
                               cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02
                if name.endswith(("proj.weight", "w_down.weight")):
                    std = 0.02 / (2 * cfg.n_layer) ** 0.5
                nn.init.normal_(p, mean=0.0, std=std)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # [snippet: forward]
    def forward(self, idx, targets=None, caches=None, start=0):
        T = idx.shape[1]
        cos = self.cos[start:start + T]       # 位置 start..start+T-1 の角度
        sin = self.sin[start:start + T]
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                   ignore_index=-100)
        return logits, loss
    # [end: forward]

    # [snippet: generate]
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=True):
        """貪欲生成。use_cache=False なら毎回すべてを計算し直す。"""
        caches = [KVCache() for _ in self.blocks] if use_cache else None
        logits, _ = self(idx, caches=caches)          # プロンプトを一度に処理
        for _ in range(max_new_tokens):
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if use_cache:                             # 新しい 1 トークンだけ
                logits, _ = self(nxt, caches=caches, start=idx.shape[1] - 1)
            else:                                     # 全部を最初から
                logits, _ = self(idx)
        return idx
    # [end: generate]
