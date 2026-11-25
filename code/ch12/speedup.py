"""高速化の定石を CPU で確かめる: SDPA、torch.compile、勾配累積。"""
import math
import time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)


def bench(fn, repeat=10):
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


# [snippet: sdpa]
B, H, T, Dh = 8, 4, 512, 48
q, k, v = (torch.randn(B, H, T, Dh) for _ in range(3))
def manual():                      # 第 9 章で書いた素朴な注意
    att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)       # (B, H, T, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return att @ v
def fused():                       # PyTorch の融合版
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
diff = (manual() - fused()).abs().max().item()
t_m, t_f = bench(manual), bench(fused)
print(f"SDPA: 素朴 {t_m * 1e3:.1f} ms, 融合 {t_f * 1e3:.1f} ms,"
      f" 最大差 {diff:.1e}")
# [end: sdpa]

# [snippet: compile]
def gelu_chain(x):                 # 要素ごとの演算が 7 回続く
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x**3)))
x = torch.randn(4096, 4096)        # 1,677 万要素 = 64 MiB
t_one = bench(lambda: x * 2.0)     # 読んで 1 回掛けて書くだけ
t_chain = bench(lambda: gelu_chain(x))
gbps = 2 * x.numel() * 4 / t_one / 1e9   # 読み 4 バイト + 書き 4 バイト
print(f"x*2: {t_one * 1e3:.1f} ms（{gbps:.1f} GB/s, "
      f"{x.numel() / t_one / 1e9:.1f} GFLOP/s）,"
      f" gelu_chain: {t_chain * 1e3:.1f} ms")
try:
    compiled = torch.compile(gelu_chain)  # 7 回の演算を 1 回に融合する
    print(f"compile 後: {bench(lambda: compiled(x)) * 1e3:.1f} ms")
except Exception as e:                    # C++ コンパイラが無いと失敗する
    print("torch.compile は使えませんでした:", type(e).__name__)
# [end: compile]

# [snippet: accum]
model = torch.nn.Linear(16, 1)
X, Y = torch.randn(32, 16), torch.randn(32, 1)
model.zero_grad()
F.mse_loss(model(X), Y).backward()            # 32 個を一度に
g_full = model.weight.grad.clone()
model.zero_grad()
for i in range(4):                            # 8 個ずつ 4 回に分ける
    xb, yb = X[8 * i:8 * i + 8], Y[8 * i:8 * i + 8]
    (F.mse_loss(model(xb), yb) / 4).backward()   # 4 で割って足し込む
diff = (model.weight.grad - g_full).abs().max().item()
print(f"勾配累積: 一括との最大差 {diff:.1e}")
# [end: accum]
