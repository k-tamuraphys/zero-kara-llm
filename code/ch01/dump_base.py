"""準備用: 基盤モデル base.pt から、第 1 章で使う実物の数値を保存する。

この章の実験は NumPy だけで行う。そのために、モデルの「最後の一歩」に
現れるベクトルと行列を、ここで一度だけ取り出してファイルにしておく。
（PyTorch とモデルの中身は第 2 章以降で説明する。この章では読まなくてよい。）

出力:
    ch01/out/last_step.npz  … h (192,), W (4096, 192), tokens (4096,)
    ch01/out/contexts.npz   … 4 つの文脈に対する次トークンの確率 (4096,)
    ch01/out/chain.txt      … "12+35=" の続きを 1 トークンずつ予測した確率
"""

from pathlib import Path

import numpy as np
import torch

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

OUT = Path(__file__).parent / "out"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()


@torch.no_grad()
def last_hidden(prompt: str) -> torch.Tensor:
    """最後の位置の隠れベクトル h（lm_head に入る直前のもの）を返す。"""
    idx = torch.tensor([tok.encode(prompt)])
    pos = torch.arange(idx.size(1))
    x = model.tok_emb(idx) + model.pos_emb(pos)
    for block in model.blocks:
        x = block(x)
    return model.ln_f(x)[0, -1]  # (192,)


@torch.no_grad()
def next_probs(prompt: str) -> torch.Tensor:
    logits, _ = model(torch.tensor([tok.encode(prompt)]))
    # 倍精度で softmax を取り、和がちょうど 1 になるようにしておく
    return torch.softmax(logits[0, -1].double(), dim=-1)  # (4096,)


# 1) 「どうも仕方が」の最後の一歩: z = W h
h = last_hidden("どうも仕方が")
W = model.lm_head.weight.detach()  # (4096, 192)
tokens = np.array([tok.decode([i]) for i in range(tok.vocab_size)])
np.savez(OUT / "last_step.npz", h=h.numpy(), W=W.numpy(), tokens=tokens)
z = W @ h
# z = W h から求めた確率が、モデルの出力と一致することを確認する
assert torch.allclose(torch.softmax(z.double(), -1),
                      next_probs("どうも仕方が"), atol=1e-6)

# 2) いくつかの文脈に対する次トークンの確率分布
contexts = {"shikata": "どうも仕方が", "drill": "\n12+35=",
            "drill47": "\n12+35=47", "wagahai": "\n　吾輩は"}
np.savez(OUT / "contexts.npz",
         **{k: next_probs(v).numpy() for k, v in contexts.items()})

# 3) 確率の連鎖律の例: "12+35=" に続く 3 トークンの確率
lines = []
for answer in ["47\n", "57\n"]:
    prompt, total = "\n12+35=", 1.0
    for ch in answer:
        p = next_probs(prompt)[tok.encode(ch)[0]].item()
        lines.append(f"p({ch!r:4} | {prompt[1:]!r:10}) = {p:.3f}")
        total *= p
        prompt += ch
    lines.append(f"  -> p({answer!r} | '12+35=') = {total:.3f}")
(OUT / "chain.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("h", tuple(h.shape), "W", tuple(W.shape))
