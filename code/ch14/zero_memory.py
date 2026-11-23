"""ZeRO の各段階で、1 GPU が持つ学習の状態（活性化を除く）を見積もる。"""

# [snippet: zero]
def per_gpu_bytes(N: float, W: int, stage: int) -> float:
    """混合精度 + AdamW。1 パラメータに重み 2・勾配 2・状態 12 バイト。"""
    p, g, o = 2 * N, 2 * N, 12 * N   # BF16 重み・BF16 勾配・FP32(主重み+m+v)
    if stage >= 1:
        o /= W                            # ZeRO-1: オプティマイザ状態を分割
    if stage >= 2:
        g /= W                            # ZeRO-2: 勾配も分割
    if stage >= 3:
        p /= W                            # ZeRO-3 (= FSDP): 重みも分割
    return p + g + o
# [end: zero]


if __name__ == "__main__":
    names = ["DDP（分割なし）", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"]
    rows = []
    for N, W in [(7.5e9, 64), (70e9, 64), (70e9, 512)]:
        print(f"N = {N/1e9:.1f}B, W = {W}")
        for s, name in enumerate(names):
            gb = per_gpu_bytes(N, W, s) / 1e9
            rows.append((N, W, s, gb))
            print(f"  {name:16s} {gb:8.1f} GB")
    with open("ch14/out/zero_memory.csv", "w") as f:
        f.write("N,W,stage,gb\n")
        for N, W, s, gb in rows:
            f.write(f"{N:.0f},{W},{s},{gb:.2f}\n")
