"""ablation.py の結果を読むための物差しを 2 つ測る（学習はしない）。

1. 同じデータ（先頭 50 万字）でのユニグラムの損失（文字の頻度だけで当てる）。
2. 学習前の「正規化なし」のモデルで、残差ストリームの値の大きさ（RMS）が
   入口から出口までにどれだけ変わるか。

実行: python ch10/ablation_baseline.py   （数秒）
出力: ch10/out/ablation_baseline.txt
"""

import math
from pathlib import Path

import torch
import torch.nn as nn

from llmbook.corpus import load_corpus
from llmbook.model import GPT, GPTConfig
from llmbook.tokenizer import CharTokenizer
from llmbook.train import get_batch

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)

text = load_corpus()[:500_000]
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
lines = []

# 1. ユニグラム: 文字 v の確率を「出た回数 / 全体」とし、平均の -log p を求める
counts = torch.bincount(data, minlength=tok.vocab_size).double()
p = counts / counts.sum()
unigram = -(p[p > 0] * p[p > 0].log()).sum().item()
lines.append(f"語彙 {tok.vocab_size} 字、"
             f"一様分布の損失 {math.log(tok.vocab_size):.3f}")
lines.append(f"ユニグラムの損失 {unigram:.3f}")

# 2. 学習前の「正規化なし」モデル（ablation.py と同じ作り方）
torch.manual_seed(0)
config = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=8,
                   n_head=4, n_embd=64)
model = GPT(config)
for block in model.blocks:
    block.ln1, block.ln2 = nn.Identity(), nn.Identity()
model.ln_f = nn.Identity()
model.apply(model._init_weights)
for name, param in model.named_parameters():
    if name.endswith("proj.weight"):
        nn.init.normal_(param, std=0.02 / math.sqrt(2 * config.n_layer))
x, _ = get_batch(data, 32, config.block_size)
with torch.no_grad():
    pos = torch.arange(x.size(1))
    h = model.tok_emb(x) + model.pos_emb(pos)
    rms_in = h.pow(2).mean().sqrt().item()
    for block in model.blocks:
        h = block(h)
    rms_out = h.pow(2).mean().sqrt().item()
lines.append(f"正規化なし（学習前）: 残差ストリームの RMS 入口 {rms_in:.4f} "
             f"-> 8 ブロック後 {rms_out:.4f}（{rms_out / rms_in:.3f} 倍）")
(OUT / "ablation_baseline.txt").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
print("\n".join(lines))
