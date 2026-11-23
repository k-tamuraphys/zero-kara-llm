"""位置エンコーディング: 原論文の正弦波の値と、位置埋め込みの効果を確かめる。

実行: python ch10/pos_encoding.py
出力: ch10/out/sinusoid.csv, pos_check.txt
"""

from pathlib import Path

import torch

from llmbook.model import GPT, GPTConfig

OUT = Path(__file__).parent / "out"
torch.manual_seed(0)


# [snippet: sinusoid]
def sinusoidal(T: int, d: int) -> torch.Tensor:
    """原論文の正弦波の位置エンコーディング。(T, d) を返す。"""
    pos = torch.arange(T).unsqueeze(1)  # (T, 1)
    i = torch.arange(0, d, 2)  # 偶数番目の次元 0, 2, 4, ...
    freq = 1.0 / (10000 ** (i / d))  # 次元が進むほど遅く回る
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * freq)  # 偶数番目は sin
    pe[:, 1::2] = torch.cos(pos * freq)  # 奇数番目は cos
    return pe
# [end: sinusoid]


pe = sinusoidal(64, 16)
with open(OUT / "sinusoid.csv", "w") as f:
    f.write("pos,d0,d2,d4,d6\n")
    for t in range(64):
        f.write(f"{t},{pe[t, 0]:.4f},{pe[t, 2]:.4f},{pe[t, 4]:.4f},"
                f"{pe[t, 6]:.4f}\n")

# [snippet: pos-check]
model = GPT(GPTConfig(vocab_size=100, block_size=8, n_layer=1,
                      n_head=2, n_embd=16)).eval()
a = torch.tensor([[5, 7, 9]])  # 「5 7 9」と
b = torch.tensor([[7, 5, 9]])  # 先頭 2 つを入れ替えた「7 5 9」
with torch.no_grad():
    za, zb = model(a)[0][0, -1], model(b)[0][0, -1]  # 最後の位置のロジット
    print("位置埋め込みあり: 差の最大", f"{(za - zb).abs().max():.2e}")
    model.pos_emb.weight.zero_()  # 位置埋め込みを 0 にして消す
    za, zb = model(a)[0][0, -1], model(b)[0][0, -1]
    print("位置埋め込みなし: 差の最大", f"{(za - zb).abs().max():.2e}")
# [end: pos-check]
