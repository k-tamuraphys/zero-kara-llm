"""図のデータ: 基盤モデル base.pt の 16 個のヘッドを、多くの問題で調べる。

2 桁 + 2 桁 = 2 桁 の足し算（例 "\n25+48=73\n"）を 300 問作り、
(1) 各ヘッドが「=」の行と、答えの十の位の行で、どの位置にどれだけ
    重みを置くかを 300 問で平均する（heads_eq.csv, heads_ans.csv）。
(2) ヘッドを 1 個ずつ止めて（そのヘッドの出力を 0 にして）、
    正しい答えの数字の確率がどれだけ下がるかを測る（heads_ablation.csv）。
"""

import csv
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.manual_seed(0)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
OUT = Path(__file__).parent / "out"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()

# 2 桁 + 2 桁 で、答えも 2 桁になる問題だけを集める（位置がそろう）
rng = random.Random(0)
lines = []
while len(lines) < 300:
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if a + b <= 99:
        lines.append(f"\n{a}+{b}={a + b}\n")
idx = torch.tensor([tok.encode(s) for s in lines])  # (300, 10)
assert idx.shape[1] == 10, "数字が 1 文字 1 トークンになっていない"
EQ, ANS = 6, 7  # 「=」の位置と、答えの十の位の位置（0 始まり）
tens_id, ones_id = idx[:, EQ + 1], idx[:, ANS + 1]  # 当てたい正解


def forward(idx, off=None):
    """ロジットと各層の注意の重みを返す。off=(層, ヘッド) を止める。"""
    B, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T))
    maps = []
    for l, block in enumerate(model.blocks):
        attn = block.attn
        h = block.ln1(x)
        q, k, v = attn.qkv(h).split(h.size(-1), dim=2)
        q, k, v = (z.view(B, T, attn.n_head, -1).transpose(1, 2)
                   for z in (q, k, v))  # (B, H, T, dh)
        S = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        S = S.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))
        A = F.softmax(S, dim=-1)
        maps.append(A)
        y = A @ v  # (B, H, T, dh)
        if off is not None and off[0] == l:
            y[:, off[1]] = 0.0  # このヘッドが運んだ情報だけを消す
        y = y.transpose(1, 2).reshape(B, T, -1)
        x = x + attn.proj(y)
        x = x + block.mlp(block.ln2(x))
    return model.lm_head(model.ln_f(x)), maps


def p_correct(logits):
    """正しい十の位（「=」の位置）と一の位（十の位の位置）の確率の平均。"""
    p = F.softmax(logits, dim=-1)
    p_tens = p[torch.arange(len(lines)), EQ, tens_id].mean().item()
    p_ones = p[torch.arange(len(lines)), ANS, ones_id].mean().item()
    return p_tens, p_ones


logits, maps = forward(idx)
assert torch.allclose(logits, model(idx)[0], atol=1e-5)  # 本物と同じ計算
base_tens, base_ones = p_correct(logits)
print(f"問題 {len(lines)} 問。止めないときの正解の確率: "
      f"十の位 {base_tens:.3f}  一の位 {base_ones:.3f}")
for name, row in [("heads_eq.csv", EQ), ("heads_ans.csv", ANS)]:
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "w"])
        for l in range(4):
            for hd in range(4):
                prof = maps[l][:, hd, row, : row + 1].mean(dim=0)
                for s in range(row + 1):
                    w.writerow([s, 4 * l + hd, f"{prof[s]:.3f}"])

with open(OUT / "heads_ablation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k", "layer", "head", "drop_tens", "drop_ones"])
    print("ヘッドを 1 個止めたときの、正解の確率の下がり幅")
    for l in range(4):
        for hd in range(4):
            p_tens, p_ones = p_correct(forward(idx, off=(l, hd))[0])
            w.writerow([4 * l + hd, l, hd, f"{base_tens - p_tens:.4f}",
                        f"{base_ones - p_ones:.4f}"])
            print(f"層{l} ヘッド{hd}: 十の位 {base_tens - p_tens:+.3f}  "
                  f"一の位 {base_ones - p_ones:+.3f}")
