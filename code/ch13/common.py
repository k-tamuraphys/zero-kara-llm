"""第 13 章の実験で共有する小道具: データの読み込みと縮小版の設定。"""

import numpy as np
import torch

from llmbook.corpus import DATA_DIR


def load_bin(name: str) -> torch.Tensor:
    """data/{name}.bin（uint16 のトークン列）を int64 のテンソルで返す。"""
    arr = np.fromfile(DATA_DIR / f"{name}.bin", dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))
