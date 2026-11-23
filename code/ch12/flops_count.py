"""ミニGPT（base.pt と同じ構成）の 1 ステップの計算量を、理論値と実測で比べる。"""
import time
import torch
from torch.utils.flop_counter import FlopCounterMode
from llmbook.model import GPT, GPTConfig

torch.manual_seed(0)
torch.set_num_threads(2)

# [snippet: theory]
def flops_per_token(V, T, L, d):
    """順伝播の 1 トークンあたりの FLOPs（掛けて足す 1 組 = 2）。"""
    qkv = 2 * d * (3 * d)        # x を Q, K, V に写す行列積
    attn = 2 * T * d * 2         # q と T 個の k の内積、T 個の v の重み付き和
    proj = 2 * d * d             # 注意の出力の射影
    mlp = 2 * d * (4 * d) * 2    # d -> 4d -> d の 2 つの行列積
    head = 2 * d * V             # 出力ヘッド（ロジット）
    return {"qkv": L * qkv, "attn": L * attn, "proj": L * proj,
            "mlp": L * mlp, "head": head}
# [end: theory]

V, T, L, d, B = 4096, 128, 4, 192, 32
cfg = GPTConfig(vocab_size=V, block_size=T, n_layer=L, n_head=4, n_embd=d)
model = GPT(cfg)
N = model.num_params()
parts = flops_per_token(V, T, L, d)
fwd = sum(parts.values())
print(f"パラメータ数 N = {N:,}")
for k, v in parts.items():
    print(f"  {k:5s}: {v:>12,} FLOPs/トークン ({v / fwd:5.1%})")
print(f"順伝播 合計 {fwd:,} FLOPs/トークン  (2N = {2 * N:,})")

# [snippet: measure]
x = torch.randint(0, V, (B, T))
y = torch.randint(0, V, (B, T))
counter = FlopCounterMode(display=False)
with counter:                       # 行列積などの FLOPs を数えてくれる
    logits, loss = model(x, y)
    loss.backward()
measured = counter.get_total_flops()
print(f"実測（順+逆）: {measured / (B * T):,.0f} FLOPs/トークン"
      f"  = 理論の順伝播の {measured / (B * T) / fwd:.2f} 倍")
# [end: measure]

# [snippet: timing]
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
times = []
for step in range(8):
    t0 = time.perf_counter()
    logits, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    times.append(time.perf_counter() - t0)
t = sorted(times[2:])[len(times[2:]) // 2]   # 最初の 2 回を捨てて中央値
print(f"1 ステップ {t * 1000:.0f} ms, 実効 {measured / t / 1e9:.1f} GFLOP/s")
# [end: timing]

# CSV: 理論の内訳
with open("ch12/out/flops_parts.csv", "w") as f:
    f.write("part,flops,share\n")
    for k, v in parts.items():
        f.write(f"{k},{v},{100 * v / fwd:.2f}\n")
with open("ch12/out/cpu_rate.txt", "w") as f:
    f.write(f"{measured / t:.4e}\n")
