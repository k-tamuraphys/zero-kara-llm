"""GPT-2 型（llmbook）と現代版ミニGPT を同じ条件で学習して比べる。

使い方: python ch11/compare_train.py gpt2   または   modern
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "ch11")
from llmbook.corpus import DATA_DIR
from llmbook.model import GPT, GPTConfig
from llmbook.train import train_lm
from modern_gpt import ModernConfig, ModernGPT

torch.set_num_threads(2)
torch.manual_seed(0)
kind = sys.argv[1]


def load(name):
    a = np.fromfile(DATA_DIR / name, dtype=np.uint16).astype(np.int64)
    return torch.from_numpy(a)


train, val = load("train.bin"), load("val.bin")
# [snippet: configs]
d, L, H = 128, 4, 4
if kind == "gpt2":
    model = GPT(GPTConfig(n_layer=L, n_head=H, n_embd=d, block_size=128))
else:
    model = ModernGPT(ModernConfig(n_layer=L, n_head=H, n_kv_head=2,
                                   n_embd=d, ffn_hidden=344, block_size=128,
                                   tie_weights=True))
# [end: configs]
print(kind, "params", f"{model.num_params():,}")
t0 = time.time()
train_lm(model, train, val, max_steps=800, batch_size=16, max_lr=2e-3,
         warmup_steps=50, eval_interval=100,
         log_path=f"ch11/out/compare_{kind}.csv")
print(f"所要 {time.time() - t0:.0f} 秒")
