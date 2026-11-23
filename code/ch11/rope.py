"""RoPE の性質を数値で確かめる。"""
import sys

import torch

sys.path.insert(0, "ch11")
from modern_gpt import apply_rope, rope_tables

torch.manual_seed(0)
# [snippet: rope-2d]
# 2 次元の例: q=(1,0), k=(1,0) を、位置 m, n に応じて回す
cos, sin = rope_tables(head_dim=2, max_len=16, theta=10000.0)  # ω=1
q = torch.tensor([1.0, 0.0])
k = torch.tensor([1.0, 0.0])
for m, n in [(0, 0), (3, 3), (5, 3), (7, 5), (3, 5)]:
    qm = apply_rope(q, cos[m], sin[m])
    kn = apply_rope(k, cos[n], sin[n])
    print(f"m={m} n={n}  m-n={m - n:+d}  q·k={(qm @ kn).item():+.4f}")
# [end: rope-2d]

# [snippet: rope-relative]
# 64 次元: 位置の差 m-n が同じなら、内積は同じになるか
dh = 64
cos, sin = rope_tables(dh, 1024)
q, k = torch.randn(dh), torch.randn(dh)
def score(m, n):
    return (apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n]))
for m, n in [(10, 7), (110, 107), (1000, 997), (10, 0), (500, 490)]:
    print(f"m={m:4d} n={n:4d} m-n={m - n:3d}  score={score(m, n).item():+.5f}")
print("回す前の q·k =", f"{(q @ k).item():+.5f}",
      " ノルム: 回す前", f"{q.norm().item():.4f}",
      "回した後", f"{apply_rope(q, cos[500], sin[500]).norm().item():.4f}")
# [end: rope-relative]
