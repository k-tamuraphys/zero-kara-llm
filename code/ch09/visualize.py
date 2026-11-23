"""学習済みの基盤モデル base.pt から注意の重みを取り出して観察する。"""

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from capture import Capture
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.manual_seed(0)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
cap = Capture()  # print の出力を out/ に保存する（capture.py）
OUT = Path(__file__).parent / "out"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()


# [snippet: get-attention]
def get_attention(model, idx):
    """各層の注意の重みを返す。リストの l 番目が (B, H, T, T)。

    CausalSelfAttention は重みを返さないので、同じ計算をここで再現する。
    """
    B, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T))
    maps = []
    for block in model.blocks:
        attn = block.attn
        h = block.ln1(x)  # ブロックの中で注意に入る値（第 10 章）
        q, k, _ = attn.qkv(h).split(h.size(-1), dim=2)
        q = q.view(B, T, attn.n_head, -1).transpose(1, 2)  # (B,H,T,dh)
        k = k.view(B, T, attn.n_head, -1).transpose(1, 2)
        S = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        S = S.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))
        maps.append(F.softmax(S, dim=-1))
        x = block(x)  # 次の層へは本物のブロックで進める
    return maps
# [end: get-attention]


def save_heatmap(name, A, labels):
    """pgfplots の matrix plot 用に x, y, 値 の 3 列で保存する。"""
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "w"])
        for i in range(len(labels)):
            for j in range(len(labels)):
                w.writerow([j, i, f"{A[i, j]:.3f}"])  # マスク部分は 0


# ---- 1. 計算ドリル「25+48=73」----
text = "\n25+48=73\n"
idx = torch.tensor([tok.encode(text)])
labels = [tok.decode([i]) for i in idx[0].tolist()]
maps = get_attention(model, idx)
logits, _ = model(idx)


def check(model, idx, maps):
    """取り出した重みで注意の出力を作り直し、本物の attn の出力と比べる。"""
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(idx.size(1)))
    err = 0.0
    for block, A in zip(model.blocks, maps):
        h = block.ln1(x)
        B, T, C = h.shape
        v = block.attn.qkv(h).split(C, dim=2)[2]
        v = v.view(B, T, block.attn.n_head, -1).transpose(1, 2)
        y = block.attn.proj((A @ v).transpose(1, 2).reshape(B, T, C))
        err = max(err, (y - block.attn(h)).abs().max().item())
        x = block(x)
    return err


print(f"本物の注意の出力との差（全層の最大）: {check(model, idx, maps):.1e}")
print("トークン:", [s.replace("\n", "\\n") for s in labels])
# [snippet: read-row]
t = labels.index("=")  # 「=」の位置。ここで答えの十の位を予測する
probs = F.softmax(logits[0, t], dim=-1)
print(f"「=」の次の予測: {tok.decode([probs.argmax().item()])!r} "
      f"(確率 {probs.max():.3f})")
for l in range(4):
    for h in range(4):
        row = maps[l][0, h, t, : t + 1]
        top = row.argmax().item()
        print(f"層{l} ヘッド{h}: 最も見ている {labels[top]!r:6} "
              f"{row[top]:.2f}")
# [end: read-row]
cap.save("drill_heads.txt")
for l, h in [(1, 0), (3, 1), (3, 3)]:
    for name in ["=", "7"]:
        t = labels.index(name)
        w = maps[l][0, h, t]
        row = "  ".join(f"{labels[s].strip() or '\\n'}:{w[s]:.2f}"
                        for s in range(t + 1) if w[s] >= 0.01)
        print(f"層{l} ヘッド{h}「{name}」の行  {row}")
print("（重みが 0.01 未満のトークンは省略）")
cap.save("drill_rows.txt")
for l, h in [(1, 0), (3, 1), (3, 3)]:
    save_heatmap(f"heat_drill_L{l}H{h}.csv", maps[l][0, h].numpy(), labels)

# ---- 2. 漱石の文 ----
text = "私はその人を常に先生と呼んでいた。"
idx = torch.tensor([tok.encode(text)])
labels2 = [tok.decode([i]) for i in idx[0].tolist()]
print("トークン:", labels2)
maps2 = get_attention(model, idx)
for l, h in [(3, 0), (2, 2)]:
    save_heatmap(f"heat_text_L{l}H{h}.csv", maps2[l][0, h].numpy(), labels2)
    for name in ["と", "んでいた。"]:
        t = labels2.index(name)
        row = " ".join(f"{labels2[s]}:{maps2[l][0, h, t, s]:.2f}"
                       for s in range(t + 1))
        print(f"層{l} ヘッド{h} の「{name}」の行: {row}")
cap.save("text_tokens.txt")

# ---- 3. 検証データ全体で「どれだけ離れたトークンを見ているか」 ----
val = torch.from_numpy(
    np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16).astype(np.int64))
starts = torch.randint(len(val) - 128, (64,))
batch = torch.stack([val[s:s + 128] for s in starts])  # (64, 128)
maps = get_attention(model, batch)
K = 16  # 距離 0（自分）〜 15 まで。位置 16 以降の行だけで平均する
prof = torch.zeros(4, 4, K)
for l in range(4):
    for k in range(K):
        # 対角線から k 本下 = 「k 個前のトークン」への重み
        diag = maps[l].diagonal(offset=-k, dim1=-2, dim2=-1)  # (64,H,T-k)
        prof[l, :, k] = diag[:, :, K - k:].mean(dim=(0, 2))
print("距離ごとの平均の重み（各層の 4 ヘッドの平均）")
print("距離   " + " ".join(f"{k:5d}" for k in range(8)))
for l in range(4):
    print(f"層{l}  " + " ".join(f"{v:5.2f}" for v in prof[l].mean(0)[:8]))
uniform = sum(1 / (t + 1) for t in range(K, 128)) / (128 - K)
print(f"参考: すべての過去に一様に配った場合は 1 個あたり平均 {uniform:.3f}")
cap.save("distance.txt")
with open(OUT / "distance.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k"] + [f"L{l}" for l in range(4)]
               + [f"L{l}H{h}" for l in range(4) for h in range(4)])
    for k in range(K):
        w.writerow([k] + [f"{prof[l, :, k].mean():.4f}" for l in range(4)]
                   + [f"{prof[l, h, k]:.4f}" for l in range(4)
                      for h in range(4)])
