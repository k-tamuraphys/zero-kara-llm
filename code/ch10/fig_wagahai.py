"""基盤モデル base.pt に「吾輩は」を入れ、次の 1 トークンを選ぶまでの
全部の段の数（形・注意の重み・上位の候補と確率）を記録する。

第 10 章のコマ割りの図（「吾輩は」の 1 トークン）の数値はこの出力から写す。
実行: python ch10/fig_wagahai.py
出力: ch10/out/fig_wagahai.txt, fig_wagahai_top.csv, fig_wagahai_steps.csv
"""

import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR, load_corpus
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

OUT = Path(__file__).parent / "out"
torch.manual_seed(0)  # 貪欲法なので乱数は使わないが、念のため固定する
torch.set_num_threads(2)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()
cfg = model.config
PROMPT = "\n　吾輩は"  # 第 1 章と同じく、行頭の改行と全角の空白を付ける
N_STEPS = 4  # 何トークン生成するか（貪欲法）
lines = []


def show(i: int) -> str:
    """トークン 1 個を表示用の文字列にする。文字の途中で切れた
    バイトは <e6> のように 16 進で示す（第 7 章のバイト単位の BPE）。"""
    if i not in tok.vocab:  # 特殊トークン
        return tok.decode([i])
    s = tok.vocab[i].decode("utf-8", errors="backslashreplace")
    s = re.sub(r"\\x([0-9a-f]{2})", r"<\1>", s)
    return s.replace("\n", "＼n")


def say(s: str = "") -> None:
    lines.append(s)


def lens_top1(x: torch.Tensor) -> tuple[str, float]:
    """途中の残差ストリームに、最後の LayerNorm と出力ヘッドを
    そのまま当てたときの、最後の位置の 1 位の候補（参考の測定）。"""
    p = F.softmax(model.lm_head(model.ln_f(x))[0, -1], dim=-1)
    j = int(p.argmax())
    return show(j), float(p[j])


def attn_last_row(block, x: torch.Tensor) -> torch.Tensor:
    """ブロックの注意の重みのうち、最後の位置の行を (H, T) で返す。
    llmbook の CausalSelfAttention と同じ計算を、同じ重みでたどり直す。"""
    B, T, C = x.shape
    H = cfg.n_head
    q, k, _ = block.attn.qkv(block.ln1(x)).split(C, dim=2)
    q = q.view(B, T, H, C // H).transpose(1, 2)  # (B, H, T, C/H)
    k = k.view(B, T, H, C // H).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(C // H)  # (B, H, T, T)
    causal = torch.tril(torch.ones(T, T)) == 0
    att = att.masked_fill(causal, float("-inf")).softmax(dim=-1)
    return att[0, :, -1, :]  # (H, T)


idx = torch.tensor([tok.encode(PROMPT)])  # (B, T) = (1, 2)
step_rows = []
with torch.no_grad():
    for step in range(1, N_STEPS + 1):
        B, T = idx.shape
        ids = idx[0].tolist()
        say(f"===== ステップ {step}（文脈 T={T}）=====")
        say("トークン: " + " | ".join(f"{show(i)}(ID {i})" for i in ids))
        say(f"入力 idx の形: {tuple(idx.shape)}")
        # 埋め込みと位置: (B, T) -> (B, T, d)
        e = model.tok_emb(idx)
        p = model.pos_emb(torch.arange(T))
        x = e + p
        say(f"トークン埋め込み {tuple(e.shape)} + 位置埋め込み "
            f"{tuple(p.shape)} -> h0 {tuple(x.shape)}")
        for l, block in enumerate(model.blocks, start=1):
            w = attn_last_row(block, x)  # (H, T)
            a = block.attn(block.ln1(x))  # 注意が足し込む量 (B, T, d)
            x = x + a
            u = F.gelu(block.mlp.fc(block.ln2(x)))  # MLP の中間 (B, T, 4d)
            x = x + block.mlp(block.ln2(x))
            word, prob = lens_top1(x)
            say(f"第 {l} ブロック: 注意の出力 {tuple(a.shape)}, "
                f"MLP の中間 {tuple(u.shape)}, 出力 {tuple(x.shape)}")
            for h in range(cfg.n_head):
                ws = ", ".join(f"{show(i)} {float(w[h, j]):.2f}"
                               for j, i in enumerate(ids))
                say(f"  ヘッド {h + 1}: 最後の位置の注意の重み [{ws}]")
            say(f"  （参考）ここで出力ヘッドに通したときの 1 位: "
                f"{word} {prob:.3f}")
        hf = model.ln_f(x)
        z = model.lm_head(hf)  # (B, T, V)
        zl = z[0, -1]  # 最後の位置のロジット (V,)
        say(f"最後の LayerNorm {tuple(hf.shape)} -> ロジット {tuple(z.shape)}"
            f" -> 最後の位置 {tuple(zl.shape)}")
        p10 = F.softmax(zl, dim=-1)  # 温度 1
        p05 = F.softmax(zl / 0.5, dim=-1)  # 温度 0.5
        top = torch.topk(zl, 8).indices.tolist()
        for r, i in enumerate(top, start=1):
            say(f"  {r} 位 {show(i)}(ID {i}): ロジット {float(zl[i]):.2f}, "
                f"確率（温度 1）{float(p10[i]):.3f}, "
                f"（温度 0.5）{float(p05[i]):.3f}")
        if step == 1:
            neko = tok.encode("猫")[0]
            rank = int((zl > zl[neko]).sum()) + 1
            say(f"  参考: 「猫」(ID {neko}) は {rank} 位、"
                f"確率（温度 1）{float(p10[neko]):.4f}")
            with open(OUT / "fig_wagahai_top.csv", "w",
                      encoding="utf-8") as f:
                f.write("rank,token,logit,p10,p05\n")
                for r, i in enumerate(top, start=1):
                    f.write(f"{r},{show(i)},{float(zl[i]):.3f},"
                            f"{float(p10[i]):.4f},{float(p05[i]):.4f}\n")
        nxt = top[0]  # 貪欲法: 確率が最大のトークンを選ぶ
        say(f"選ぶ（貪欲法）: {show(nxt)}(ID {nxt}) "
            f"バイト列 {tok.vocab[nxt].hex(' ')}")
        step_rows.append((step, T, show(top[0]), float(p10[top[0]]),
                          show(top[1]), float(p10[top[1]]),
                          show(top[2]), float(p10[top[2]])))
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)  # 末尾に付ける
        say()
    new = idx[0, len(tok.encode(PROMPT)):].tolist()
    say("生成したトークン: " + " | ".join(show(i) for i in new))
    say("つなげて文字にすると: 吾輩は" + tok.decode(new))

# 参考: コーパスで「吾輩は」の直後に何が続くかを数える（モデルとの比較用）
text = load_corpus()
say(f"コーパスの「吾輩は」: {text.count('吾輩は')} 回。直後が「猫」"
    f"{text.count('吾輩は猫')} 回、「主人」{text.count('吾輩は主人')} 回、"
    f"「この」{text.count('吾輩はこの')} 回")

with open(OUT / "fig_wagahai_steps.csv", "w", encoding="utf-8") as f:
    f.write("step,T,top1,p1,top2,p2,top3,p3\n")
    for row in step_rows:
        s, T, a, pa, b, pb, c, pc = row
        f.write(f"{s},{T},{a},{pa:.4f},{b},{pb:.4f},{c},{pc:.4f}\n")
(OUT / "fig_wagahai.txt").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
print("\n".join(lines))
