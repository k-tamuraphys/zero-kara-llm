"""RoPE の周波数と、距離による内積の変化を CSV に出す（図用）。"""
import sys

import torch

sys.path.insert(0, "ch11")
from modern_gpt import apply_rope, rope_tables

dh = 64
cos, sin = rope_tables(dh, 512)
q = torch.ones(dh)  # 全成分 1 の q と k（同じ向き）
with open("ch11/out/rope_decay.csv", "w") as f:
    f.write("dist,score\n")
    for dist in range(0, 257, 2):
        s = apply_rope(q, cos[dist], sin[dist]) @ apply_rope(q, cos[0], sin[0])
        f.write(f"{dist},{s.item() / dh:.5f}\n")
i = torch.arange(dh // 2)
omega = 10000.0 ** (-2 * i / dh)
with open("ch11/out/rope_freq.csv", "w") as f:
    f.write("i,period\n")
    for j in range(dh // 2):
        f.write(f"{j},{(2 * torch.pi / omega[j]).item():.3f}\n")
print(f"周期 i=0: {(2 * torch.pi / omega[0]).item():.2f}  "
      f"i=31: {(2 * torch.pi / omega[-1]).item():.0f}")
