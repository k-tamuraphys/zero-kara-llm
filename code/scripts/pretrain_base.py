"""ミニGPT の事前学習。以降の章で使う基盤モデル base.pt を作る。

実行:  python scripts/pretrain_base.py   （先に prepare_data.py を実行）
CPU (4 コア) で 30〜40 分、GPU なら 1〜2 分程度。
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.arith import drill_accuracy
from llmbook.corpus import DATA_DIR
from llmbook.model import GPT, GPTConfig
from llmbook.tokenizer import BPETokenizer
from llmbook.train import save_checkpoint, train_lm

CKPT_DIR = DATA_DIR.parent / "checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
train_data = torch.from_numpy(
    np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16).astype(np.int64))
val_data = torch.from_numpy(
    np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16).astype(np.int64))

config = GPTConfig(vocab_size=tok.vocab_size, block_size=128,
                   n_layer=4, n_head=4, n_embd=192)
model = GPT(config)
print(f"パラメータ数: {model.num_params():,}")

train_lm(model, train_data, val_data, max_steps=4000, batch_size=32,
         max_lr=1e-3, warmup_steps=200, eval_interval=500, device=device,
         log_path=CKPT_DIR / "base_log.csv")
save_checkpoint(model, CKPT_DIR / "base.pt")

# 学習結果を確かめる: 小説の続きを書かせ、計算ドリルの正答率を測る
model.eval()
prompt = torch.tensor([tok.encode("\n　吾輩は")], device=device)
out = model.generate(prompt, 80, temperature=0.8, top_k=50)
print(tok.decode(out[0].tolist()))
print(f"計算ドリル正答率: {drill_accuracy(model, tok, device=device):.1%}")
