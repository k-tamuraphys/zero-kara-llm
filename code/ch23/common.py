"""第 23 章のスクリプトが共有する小さな道具。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer

CKPT_DIR = DATA_DIR.parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "out"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)  # CPU を他の作業と分け合う

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
pad_id = tok.special_tokens["<|pad|>"]
