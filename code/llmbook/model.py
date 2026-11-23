"""ミニGPT: デコーダのみの Transformer 言語モデル（GPT-2 と同じ構造）。"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# [snippet: config]
@dataclass
class GPTConfig:
    vocab_size: int = 4096  # 語彙の大きさ V
    block_size: int = 128  # 一度に見られるトークン数（文脈長）T
    n_layer: int = 4  # Transformer ブロックの数 L
    n_head: int = 4  # 注意ヘッドの数 H
    n_embd: int = 128  # 埋め込みの次元 d
    dropout: float = 0.0  # ドロップアウト率
# [end: config]


# [snippet: attention]
class CausalSelfAttention(nn.Module):
    """因果マスク付きのマルチヘッド自己注意。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 未来を見ないための下三角マスク（学習されない定数）
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, *mask.shape),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # バッチ, 系列長, 埋め込み次元
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, C/H): ヘッドごとに分ける
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 注意スコア: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, H, T, C/H)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # ヘッドを結合
        return self.resid_dropout(self.proj(y))
# [end: attention]


# [snippet: mlp]
class MLP(nn.Module):
    """位置ごとの全結合ネットワーク（フィードフォワード層）。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))
# [end: mlp]


# [snippet: block]
class Block(nn.Module):
    """Transformer ブロック: 注意と MLP を残差接続でつなぐ（Pre-LN）。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
# [end: block]


# [snippet: gpt-init]
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 重み共有 (weight tying)

        self.apply(self._init_weights)
        # 残差に足し込む直前の層は、層数に応じて小さく初期化する。
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                std = 0.02 / math.sqrt(2 * config.n_layer)
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
# [end: gpt-init]

    # [snippet: gpt-forward]
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) のトークン ID。logits (B, T, V) と損失を返す。

        targets の値が -100 の位置は損失の計算から除外される。
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "系列が文脈長を超えている"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss
    # [end: gpt-forward]

    # [snippet: generate]
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, stop_token: int | None = None):
        """idx (B, T) の続きを 1 トークンずつサンプリングして伸ばす。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # 文脈長に切り詰める
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 最後の位置の予測だけを使う
            if temperature == 0.0:  # 温度 0 は「最も確率の高いトークンを選ぶ」
                next_id = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                if stop_token is not None and (next_id == stop_token).all():
                    break
                continue
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[:, [-1]]  # k 番目の値
                logits[logits < kth] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                # 累積確率が top_p を超えた後ろのトークンを捨てる
                remove = probs.cumsum(dim=-1) - probs > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx
    # [end: generate]
