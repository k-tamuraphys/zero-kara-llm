"""残差接続と LayerNorm を外すと、深いモデルの学習はどうなるか（8 層）。

実行: python ch10/ablation.py   （2 スレッドで約 4 分）
出力: ch10/out/ablation.txt, ablation.csv
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from llmbook.corpus import load_corpus
from llmbook.model import GPT, Block, GPTConfig
from llmbook.tokenizer import CharTokenizer
from llmbook.train import get_batch, make_optimizer

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)


# [snippet: variants]
class NoResidualBlock(Block):
    """残差接続を外したブロック: 足し込まずに置き換える。"""
    def forward(self, x):
        x = self.attn(self.ln1(x))
        return self.mlp(self.ln2(x))
# [end: variants]


def make_model(kind: str, config: GPTConfig) -> GPT:
    model = GPT(config)
    for i in range(config.n_layer):
        if kind == "no_residual":
            model.blocks[i] = NoResidualBlock(config)
        if kind == "no_layernorm":  # 正規化を「何もしない」に置き換える
            model.blocks[i].ln1 = nn.Identity()
            model.blocks[i].ln2 = nn.Identity()
    if kind == "no_layernorm":
        model.ln_f = nn.Identity()
    model.apply(model._init_weights)  # 差し替えた層も同じ規則で初期化
    for name, p in model.named_parameters():
        if name.endswith("proj.weight"):
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layer))
    return model


text = load_corpus()[:500_000]
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
config = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=8,
                   n_head=4, n_embd=64)
kinds = ["full", "no_residual", "no_layernorm"]
curves = {}
for kind in kinds:
    torch.manual_seed(0)
    model = make_model(kind, config)
    opt = make_optimizer(model, 1e-3)
    curves[kind], t0 = [], time.time()
    for step in range(600):
        x, y = get_batch(data, 32, config.block_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curves[kind].append(loss.item())
    last = sum(curves[kind][-50:]) / 50
    print(f"{kind:<13s} 最後の 50 ステップの平均損失 {last:.3f} "
          f"({time.time() - t0:.0f} 秒)")
with open(OUT / "ablation.csv", "w") as f:
    f.write("step," + ",".join(kinds) + "\n")
    for s in range(0, 600, 10):  # 10 ステップごとに、前後 10 個の平均
        vals = [sum(curves[k][s:s + 10]) / 10 for k in kinds]
        f.write(f"{s + 5}," + ",".join(f"{v:.4f}" for v in vals) + "\n")
