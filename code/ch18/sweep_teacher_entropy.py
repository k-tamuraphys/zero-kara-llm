"""第 18 章: 教師の分布が「どのくらい迷っているか」を、課題ごとに測る。

計算の対話（教師 sft.pt）の答えの位置と、漱石の文章（教師 base.pt）の
各位置で、教師の分布のエントロピーと、いちばん高い確率を集計する。
python ch18/sweep_teacher_entropy.py > ch18/out/sweep_teacher_entropy.txt
"""
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F

from llmbook.chat import IGNORE, encode_chat_example, make_chat_example
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
torch.manual_seed(0)
CK = DATA_DIR.parent / "checkpoints"
OUT = DATA_DIR.parent / "ch18" / "out"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")


def stats(logits: torch.Tensor):
    """ロジット (N, V) から、各位置のエントロピーと最大確率を返す。"""
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)           # 自然対数（ナット）
    return ent, logp.exp().max(-1).values


rows = []
# 1. 計算の対話: 答えの位置を「数字の位置」と「それ以外」に分ける
sft = load_checkpoint(CK / "sft.pt").eval()
rng = random.Random(2024)
ent_d, top_d, ent_o, top_o = [], [], [], []
with torch.no_grad():
    for _ in range(500):
        msgs, _ = make_chat_example(rng)
        ids, labels = encode_chat_example(tok, msgs)
        z = sft(torch.tensor([ids]))[0][0]       # (T, V)
        y = torch.tensor(labels)
        mask = y != IGNORE
        ent, top = stats(z[mask])
        for e, t, lab in zip(ent.tolist(), top.tolist(), y[mask].tolist()):
            is_digit = any(c.isdigit() for c in tok.decode([lab]))
            (ent_d if is_digit else ent_o).append(e)
            (top_d if is_digit else top_o).append(t)
n_all = len(ent_d) + len(ent_o)
for name, ent, top in (("drill_digit", ent_d, top_d),
                       ("drill_other", ent_o, top_o)):
    rows.append(dict(kind=name, n=len(ent), share=len(ent) / n_all,
                     mean_ent=float(np.mean(ent)),
                     frac_top099=float(np.mean(np.array(top) > 0.99))))

# 2. 漱石の文章: 検証データの窓のすべての位置
base = load_checkpoint(CK / "base.pt").eval()
a = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16)
val = torch.from_numpy(a[:int(np.argmax(a == 4088))].astype(np.int64))
g = torch.Generator().manual_seed(123)
ent_s, top_s = [], []
with torch.no_grad():
    for _ in range(5):
        i = torch.randint(len(val) - 129, (32,), generator=g)
        x = torch.stack([val[j:j + 128] for j in i])   # (32, 128)
        ent, top = stats(base(x)[0].flatten(0, 1))
        ent_s += ent.tolist()
        top_s += top.tolist()
rows.append(dict(kind="soseki", n=len(ent_s), share=1.0,
                 mean_ent=float(np.mean(ent_s)),
                 frac_top099=float(np.mean(np.array(top_s) > 0.99))))

for r in rows:
    print(f"{r['kind']:12s} positions {r['n']:6d}  share {r['share']:.3f}  "
          f"mean entropy {r['mean_ent']:.3f} nats  "
          f"top-1 > 0.99: {r['frac_top099']:.3f}")
with open(OUT / "sweep_teacher_entropy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
