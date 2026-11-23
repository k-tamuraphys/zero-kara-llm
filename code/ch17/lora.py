"""第 17 章: LoRA（低ランク適応）を nn.Linear を包むクラスとして自作する。"""

import math

import torch
import torch.nn as nn


# [snippet: lora-linear]
class LoRALinear(nn.Module):
    """凍結した線形層 W に、学習する低ランクの差分 (alpha/r) B A を足す。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base                      # 元の層（W は凍結する）
        self.base.weight.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in))   # (r, d_in)
        self.B = nn.Parameter(torch.zeros(d_out, r))  # (d_out, r) は 0 始まり
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # nn.Linear と同じ
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)                      # x W^T     (B, T, d_out)
        if not self.merged:                   # + x A^T B^T (同じ形)
            y = y + (x @ self.A.T @ self.B.T) * self.scale
        return y
# [end: lora-linear]

    # [snippet: lora-merge]
    @torch.no_grad()
    def merge(self) -> None:
        """W <- W + (alpha/r) B A。以後は元の行列積 1 回だけで済む。"""
        if not self.merged:
            self.base.weight += self.scale * (self.B @ self.A)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """マージを取り消し、元の W に戻す（アダプタの付け替え用）。"""
        if self.merged:
            self.base.weight -= self.scale * (self.B @ self.A)
            self.merged = False
    # [end: lora-merge]


# [snippet: add-lora]
TARGETS_ALL = ("attn.qkv", "attn.proj", "mlp.fc", "mlp.proj")


def add_lora(model: nn.Module, r: int = 8, alpha: float = 16.0,
             targets=TARGETS_ALL) -> nn.Module:
    """名前が targets で終わる nn.Linear を LoRALinear に差し替える。"""
    for p in model.parameters():          # まず全パラメータを凍結する
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full.endswith(targets):
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return model
# [end: add-lora]


# [snippet: lora-state]
def lora_state_dict(model: nn.Module) -> dict:
    """アダプタ（A と B）だけを取り出す。保存するのはこれだけでよい。"""
    return {k: v for k, v in model.state_dict().items()
            if k.endswith((".A", ".B"))}


def count_params(model: nn.Module) -> tuple[int, int]:
    """(学習するパラメータ数, 全パラメータ数)。重み共有は 1 回だけ数える。"""
    params = {id(p): p for p in model.parameters()}.values()
    n_train = sum(p.numel() for p in params if p.requires_grad)
    return n_train, sum(p.numel() for p in params)
# [end: lora-state]


def merge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()
