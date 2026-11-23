"""SFT の条件を変えて比べる実験。

実行:  python ch16/sft_experiments.py data base     データ量（事前学習あり）
       python ch16/sft_experiments.py data scratch  データ量（事前学習なし）
       python ch16/sft_experiments.py lr       学習率（約 6 分）
       python ch16/sft_experiments.py mask     損失マスクの有無（約 3 分）
出力:  ch16/out/sft_<実験名>.csv と .txt
"""

import csv
import random
import sys
import time

import torch

from common import (OUT_DIR, device, load_base, run_sft, soseki_val_loss,
                    tok)
from llmbook.arith import drill_accuracy
from llmbook.chat import chat_accuracy, encode_chat_example, make_chat_example
from llmbook.model import GPT

end_id = tok.special_tokens["<|end|>"]


# [snippet: from-scratch]
def scratch_model():
    """base.pt と同じ形で、重みが乱数のままのモデル（事前学習なし）。"""
    torch.manual_seed(0)
    return GPT(load_base().config).to(device)
# [end: from-scratch]


def make_examples(n, seed=0, learn_all=False):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ids, labels = encode_chat_example(tok, make_chat_example(rng)[0])
        if learn_all:  # 損失マスクなし: すべての位置で次トークンを当てる
            labels = ids[1:] + [end_id]  # 最後の正解は必ず <|end|>
        out.append((ids, labels))
    return out


def save(name, rows):
    with open(OUT_DIR / f"sft_{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


experiment = sys.argv[1] if len(sys.argv) > 1 else "data"
t0 = time.time()
rows = []
if experiment == "data":
    init = sys.argv[2]  # "base"（事前学習あり）か "scratch"（なし）
    experiment += "_" + init  # 出力は sft_data_base.csv などに分ける
    # [snippet: exp-data]
    for n in [30, 100, 300, 1000, 3000, 10000]:
        examples = make_examples(n)  # 種 0 なので、少ない側は多い側の先頭
        model = load_base() if init == "base" else scratch_model()
        run_sft(model, examples)  # どの n でも同じ 600 ステップ
        acc = chat_accuracy(model, tok, device=device)
        rows.append({"n": n, "init": init, "chat_acc": acc})
        print(rows[-1], f"{time.time() - t0:.0f} 秒", flush=True)
    # [end: exp-data]
elif experiment == "lr":
    examples = make_examples(3000)
    for max_lr in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
        model = run_sft(load_base(), examples, max_lr=max_lr)
        rows.append({"lr": max_lr,
                     "chat_acc": chat_accuracy(model, tok, device=device),
                     "soseki_loss": soseki_val_loss(model),
                     "drill_acc": drill_accuracy(model, tok, device=device)})
        print(rows[-1], f"{time.time() - t0:.0f} 秒", flush=True)
elif experiment == "mask":
    for learn_all in [False, True]:
        model = run_sft(load_base(), make_examples(3000, learn_all=learn_all))
        rows.append({"learn_all": learn_all,
                     "chat_acc": chat_accuracy(model, tok, device=device),
                     "soseki_loss": soseki_val_loss(model)})
        print(rows[-1], f"{time.time() - t0:.0f} 秒", flush=True)
        # <|user|> だけを与えると、モデルは何を書くか（質問を作り出すか）
        idx = torch.tensor([tok.encode("<|user|>")], device=device)
        torch.manual_seed(0)
        out = model.generate(idx, 12, temperature=0.8, top_k=50)
        rows[-1]["after_user"] = tok.decode(out[0, 1:].tolist())
save(experiment, rows)
print(f"所要時間 {time.time() - t0:.0f} 秒")
