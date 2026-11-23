"""密な FFN と MoE（負荷分散損失の有無）を同じ条件で学習して比べる。

使い方: python ch11/moe_experiment.py dense | moe | moe_noaux
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "ch11")
from llmbook.corpus import DATA_DIR
from llmbook.train import estimate_loss, get_batch, lr_at
from modern_gpt import ModernConfig, ModernGPT

torch.set_num_threads(1)
torch.manual_seed(0)
kind = sys.argv[1]
load = lambda n: torch.from_numpy(
    np.fromfile(DATA_DIR / n, dtype=np.uint16).astype(np.int64))
train, val = load("train.bin"), load("val.bin")

# [snippet: moe-config]
cfg = ModernConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=96,
                   ffn_hidden=256, block_size=64, tie_weights=True,
                   n_expert=0 if kind == "dense" else 4, top_k=1)
aux_coef = 0.01 if kind == "moe" else 0.0
# [end: moe-config]
model = ModernGPT(cfg)
n_total = model.num_params()
ffn_one = sum(p.numel() for p in model.blocks[0].ffn.parameters())
print(f"{kind}: 総パラメータ {n_total:,}  1 層の FFN {ffn_one:,}")
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95),
                        weight_decay=0.1)
steps, t0 = 800, time.time()
log = open(f"ch11/out/moe_{kind}.csv", "w")
log.write("step,val_loss,max_share\n")
# [snippet: moe-loop]
for step in range(steps + 1):
    if step % 100 == 0:
        share = 0.0
        if cfg.n_expert:
            u = sum(b.ffn.usage for b in model.blocks)
            share = (u / u.sum()).max().item()   # 最も混んだエキスパートの割合
        v = estimate_loss(model, val, 16)
        log.write(f"{step},{v:.4f},{share:.3f}\n")
        print(f"step {step:4d} val {v:.3f} 最大の負荷 {share:.2f}")
    if step == steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step, 2e-3, 50, steps)
    x, y = get_batch(train, 16, cfg.block_size)
    _, loss = model(x, y)
    if cfg.n_expert:                              # 負荷分散損失を足す
        loss = loss + aux_coef * sum(b.ffn.aux_loss for b in model.blocks)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
# [end: moe-loop]
if cfg.n_expert:
    for i, b in enumerate(model.blocks):
        u = b.ffn.usage
        print(f"層 {i} の各エキスパートの割合:",
              " ".join(f"{x:.2f}" for x in (u / u.sum()).tolist()))
torch.manual_seed(1)                              # 最後は 100 バッチで丁寧に
final = estimate_loss(model, val, 16, 100)
print(f"最終の検証損失（100 バッチ）: {final:.3f}")
print(f"所要 {time.time() - t0:.0f} 秒")
