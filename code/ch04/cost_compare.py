"""順伝播と逆伝播の時間を、基盤モデル base.pt（約 260 万パラメータ）で測る。

数値微分で全パラメータの勾配を求めると何回の順伝播が要るか、も見積もる。
共有 CPU なので、各 7 回測って中央値をとる。
"""

import statistics
import time

import torch

from llmbook.corpus import DATA_DIR
from llmbook.train import load_checkpoint

torch.manual_seed(0)
torch.set_num_threads(2)
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
n_params = model.num_params()
idx = torch.randint(0, 4096, (8, 128))  # トークン 8 × 128 個（中身は乱数）
tgt = torch.randint(0, 4096, (8, 128))

fwd, bwd = [], []
for _ in range(8):
    t0 = time.perf_counter()
    _, loss = model(idx, tgt)  # 順伝播（損失まで）
    t1 = time.perf_counter()
    model.zero_grad()
    loss.backward()  # 逆伝播（全パラメータの勾配）
    t2 = time.perf_counter()
    fwd.append(t1 - t0)
    bwd.append(t2 - t1)
f = statistics.median(fwd[1:])  # 1 回目は準備が入るので捨てる
b = statistics.median(bwd[1:])
print(f"パラメータ数           {n_params:,}")
print(f"順伝播 1 回            {f * 1000:.0f} ミリ秒")
ratio = b / f
print(f"逆伝播 1 回            {b * 1000:.0f} ミリ秒"
      f"（順伝播の {ratio:.1f} 倍）")
days = (n_params + 1) * f / 86400
print(f"数値微分（順伝播 {n_params + 1:,} 回）  約 {days:.1f} 日")

# Transformer の中の MLP が占める割合
mlp = sum(p.numel() for blk in model.blocks for p in blk.mlp.parameters())
share = mlp / n_params
print(f"ブロック内の MLP のパラメータ数  {mlp:,}（全体の {share:.0%}）")
