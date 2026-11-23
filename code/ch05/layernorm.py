"""第 5 章: レイヤー正規化を自作し、nn.LayerNorm と比べる。

    cd code && python ch05/layernorm.py > ch05/out/layernorm.txt
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


# [snippet: layernorm]
class LayerNorm(nn.Module):
    """レイヤー正規化。データ 1 件ごとに、平均 0・分散 1 にそろえる。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))    # 倍率（1 から学習）
        self.beta = nn.Parameter(torch.zeros(dim))    # ずらす量（0 から学習）
        self.eps = eps

    def forward(self, u):                             # u: (..., dim)
        mean = u.mean(dim=-1, keepdim=True)           # (..., 1) 1 件ごとの平均
        var = ((u - mean) ** 2).mean(dim=-1, keepdim=True)   # 1 件ごとの分散
        u_hat = (u - mean) / torch.sqrt(var + self.eps)      # 平均 0・分散 1
        return self.gamma * u_hat + self.beta
# [end: layernorm]


# 手計算の例
u = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
ln = LayerNorm(4)
print("入力      :", u[0].tolist())
print("平均・分散:", u.mean().item(), ((u - u.mean()) ** 2).mean().item())
print("出力      :", [round(v, 3) for v in ln(u)[0].tolist()])
print("10 倍の入力:", [round(v, 3) for v in ln(10 * u)[0].tolist()])
print("---")

# nn.LayerNorm と比べる
x = torch.randn(300, 32) * 5 + 3                     # 大きさも中心もずれた入力
mine, ref = LayerNorm(32), nn.LayerNorm(32)
y = mine(x)
y = y.detach()
print(f"入力の 1 件目: 平均 {x[0].mean():.3f}"
      f"  標準偏差 {x[0].std(unbiased=False):.3f}")
print(f"出力の 1 件目: 平均 {y[0].mean():.3f}"
      f"  標準偏差 {y[0].std(unbiased=False):.3f}")
print(f"nn.LayerNorm との差の最大: {(y - ref(x)).abs().max():.1e}")
print("パラメータ数:", sum(p.numel() for p in mine.parameters()))
print("---")

# バッチ正規化は、同じバッチに入ったほかのデータの影響を受ける
bn, ln3 = nn.BatchNorm1d(3), nn.LayerNorm(3)
a = torch.tensor([1.0, 2.0, 6.0])                    # 注目するデータ
others1 = torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
others2 = torch.tensor([[9.0, 0.0, 5.0], [4.0, 8.0, 0.0]])
for name, others in (("仲間 A", others1), ("仲間 B", others2)):
    batch = torch.cat([a[None], others])             # (3, 3)
    print(f"{name} と組んだとき")
    print("  バッチ正規化  :", [round(v, 3) for v in bn(batch)[0].tolist()])
    print("  レイヤー正規化:", [round(v, 3) for v in ln3(batch)[0].tolist()])
