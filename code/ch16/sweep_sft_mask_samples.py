"""損失マスクの有無で、<|user|> の後に何を書くかを 20 回ずつ確かめる。

表 16.4 の右端の列は、温度 0.8 で 1 回だけ書かせた例だった。ここでは
sft_experiments.py の mask 実験と同じ 2 つのモデルを作り直し、乱数の種を
0〜19 に変えて 20 回ずつ書かせ、「質問の言い方」が現れた回数を数える。
種 0 の結果は、表 16.4 の例と同じになるはず。
実行:  python ch16/sweep_sft_mask_samples.py \
           > ch16/out/sweep_sft_mask_samples.txt
出力:  ch16/out/sweep_sft_mask_samples.csv     CPU で約 3 分。
"""

import csv
import random

import torch

from common import OUT_DIR, device, load_base, run_sft, tok
from llmbook.chat import encode_chat_example, make_chat_example

end_id = tok.special_tokens["<|end|>"]
# 質問の言い方（llmbook/chat.py の QUESTION_TEMPLATES）に出てくる語
QUESTION_WORDS = ["は？", "足す", "引く", "いくつ"]


def make_examples(n, seed=0, learn_all=False):
    """sft_experiments.py の make_examples と同じもの。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ids, labels = encode_chat_example(tok, make_chat_example(rng)[0])
        if learn_all:  # 損失マスクなし: すべての位置で次トークンを当てる
            labels = ids[1:] + [end_id]
        out.append((ids, labels))
    return out


rows = []
for learn_all in [False, True]:
    model = run_sft(load_base(), make_examples(3000, learn_all=learn_all))
    model.eval()
    name = "マスクなし" if learn_all else "マスクあり"
    n_question = 0
    for seed in range(20):
        idx = torch.tensor([tok.encode("<|user|>")], device=device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(idx, 12, temperature=0.8, top_k=50)
        text = tok.decode(out[0, 1:].tolist())
        is_q = any(w in text for w in QUESTION_WORDS)
        n_question += is_q
        rows.append({"model": name, "seed": seed, "question": int(is_q),
                     "text": text})
        print(f"{name} 種 {seed:2d} {'質問' if is_q else '    '} {text!r}")
    print(f"{name}: 20 回のうち質問の言い方が現れたのは {n_question} 回")

with open(OUT_DIR / "sweep_sft_mask_samples.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
