"""基盤モデルのパラメータ数を式で数え、コードの値と比べる。形も追跡する。

実行: python ch10/count_params.py
出力: ch10/out/count_params.txt, count_modules.txt, shapes.txt,
      param_share.csv
"""

from pathlib import Path

import torch

from llmbook.model import GPT, GPTConfig

OUT = Path(__file__).parent / "out"
torch.manual_seed(0)

# [snippet: count]
cfg = GPTConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4,
                n_embd=192)  # 基盤モデル base.pt と同じ設定
V, T, d, L = cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_layer
by_formula = {
    "tok_emb      V*d": V * d,
    "pos_emb      T*d": T * d,
    "block x 1    12d^2+4d": 12 * d * d + 4 * d,
    "block x L    L(12d^2+4d)": L * (12 * d * d + 4 * d),
    "ln_f         2d": 2 * d,
    "lm_head      0 (tied)": 0,
}
total = V * d + T * d + L * (12 * d * d + 4 * d) + 2 * d
model = GPT(cfg)
lines = [f"{k:<24s}{v:>12,}" for k, v in by_formula.items()]
lines += [f"{'total (formula)':<24s}{total:>12,}",
          f"{'model.num_params()':<24s}{model.num_params():>12,}"]
# [end: count]
(OUT / "count_params.txt").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
print("\n".join(lines))

# [snippet: count-modules]
lines = []
for name, p in model.named_parameters():
    if name.startswith("blocks.") and not name.startswith("blocks.0."):
        continue  # ブロック 1〜3 はブロック 0 と同じなので省く
    lines.append(f"{name:<24s}{str(tuple(p.shape)):>12s}{p.numel():>10,}")
lines.append(f"lm_head と tok_emb は同じテンソルか: "
             f"{model.lm_head.weight is model.tok_emb.weight}")
# [end: count-modules]
(OUT / "count_modules.txt").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
print("\n".join(lines))

# [snippet: shapes]
shapes = []
def hook(name):
    return lambda mod, inp, out: shapes.append(
        f"{name:<14s}{str(tuple(inp[0].shape)):>16s} -> "
        f"{str(tuple(out.shape)):<16s}")
for name in ["tok_emb", "pos_emb", "blocks.0.ln1", "blocks.0.attn",
             "blocks.0.mlp.fc", "blocks.0.mlp", "blocks.0", "ln_f",
             "lm_head"]:
    model.get_submodule(name).register_forward_hook(hook(name))
idx = torch.randint(0, V, (2, 8))  # B=2, T=8 の入力
logits, _ = model(idx)
# [end: shapes]
(OUT / "shapes.txt").write_text("\n".join(shapes[:9]) + "\n",
                                encoding="utf-8")
print("\n".join(shapes[:9]))


# [snippet: share]
def shares(V, T, L, d):
    """式で数えた内訳（バイアスなし）を、全体に対する割合 (%) で返す。"""
    parts = [V * d, T * d, L * 4 * d * d, L * 8 * d * d, L * 4 * d + 2 * d]
    return sum(parts), [100 * n / sum(parts) for n in parts]
# [end: share]


with open(OUT / "param_share.csv", "w", encoding="utf-8") as f:
    f.write("model,total,emb,pos,attn,mlp,ln\n")
    for name, args in [("base", (4096, 128, 4, 192)),
                       ("gpt2-xl", (50257, 1024, 48, 1600))]:
        n, pct = shares(*args)
        f.write(f"{name},{n}," + ",".join(f"{x:.1f}" for x in pct) + "\n")
print((OUT / "param_share.csv").read_text(encoding="utf-8"), end="")
