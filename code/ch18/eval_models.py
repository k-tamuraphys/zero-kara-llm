"""第 18 章: 教師と生徒の正答率（2 桁まで）と、応答の速さを比べる。"""
import time

import torch

from llmbook.chat import chat_accuracy, chat_reply
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
CK = DATA_DIR.parent / "checkpoints"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
teacher = load_checkpoint(CK / "sft.pt").eval()
student = load_checkpoint(CK / "ch18_hard_n3000_mv99.pt").eval()
acc = chat_accuracy(teacher, tok, 1000, seed=777, max_value=99)
print(f"teacher accuracy (max_value=99, 1000, seed 777): {acc:.3f}")
for name, m in (("teacher", teacher), ("student", student)):
    times = []
    for _ in range(3):                    # 混雑でぶれるので 3 回測る
        t0 = time.perf_counter()
        for _ in range(100):
            chat_reply(m, tok, "12+35は？")
        times.append((time.perf_counter() - t0) / 100 * 1000)
    print(f"{name}: {m.num_params():,} params, "
          f"ms/reply = " + ", ".join(f"{t:.1f}" for t in times))
