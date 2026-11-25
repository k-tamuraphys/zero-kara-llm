"""第 25 章：縮小版の大きさと事前学習の長さを変えて、各段の値を測る。

  python ch25/sweep_scale.py 設定名   # 1 つの設定を走らせる（2〜6 分）
  python ch25/sweep_scale.py combine  # 結果を 2 つの CSV にまとめる
     設定名: e48 / e96 / e192（次元）、s300 / s1200（事前学習のステップ数）
出力: ch25/out/sweep_scale_設定名.csv、sweep_scale_size.csv、
      sweep_scale_steps.csv
pipeline.py の関数をそのまま使い、強化学習と量子化の段は省く。
e96 は pipeline.py の縮小版と同じ設定なので、同じ値が出るはずである。
"""
import csv
import random
import re
import sys
import time
from pathlib import Path

import torch

import pipeline
from pipeline import (CFG, OUT, stage_data, stage_tokenizer, stage_pretrain,
                      stage_sft, acc)
from llmbook.arith import complete, drill_accuracy, make_problem
from llmbook.model import GPTConfig

RUNS = {  # 設定名: (n_embd, 事前学習のステップ数)。層とヘッドは 2 のまま
    "e48": (48, 600), "e96": (96, 600), "e192": (192, 600),
    "s300": (96, 300), "s1200": (96, 1200),
}
TMP = Path("/tmp/ch25_sweep")             # 途中の学習曲線の置き場
DRILL = re.compile(r"^\d+[+-]\d+=\d+$")   # ドリルの行の形


@torch.no_grad()
def split_loss(model, tok, val_text):
    """検証データを小説の行とドリルの行に分け、それぞれの平均損失を返す。"""
    model.eval()
    T = model.config.block_size
    res = []
    for want in (False, True):             # 小説の行、ドリルの行の順
        lines = [s for s in val_text.split("\n")
                 if s and bool(DRILL.match(s)) == want]
        data = torch.tensor(tok.encode("\n".join(lines)))
        n = (len(data) - 1) // T           # 重ならない窓をすべて使う
        x = torch.stack([data[i * T:(i + 1) * T] for i in range(n)])
        y = torch.stack([data[i * T + 1:(i + 1) * T + 1] for i in range(n)])
        _, loss = model(x, y)
        res.append((loss.item(), len(data)))
    return res


def drill_eq(model, tok, n=200, seed=12345):
    """「=」を含めずに問題を与え、「=答え」と続けられる割合を返す。

    語彙 512 では「=4」が 1 トークンになるので、「=」で終わる問題文は
    学習中に一度も見ない切れ目になる（drill_accuracy はそう与える）。
    """
    rng = random.Random(seed)              # drill_accuracy と同じ問題
    qs = [make_problem(rng, CFG["max_value"], "+-") for _ in range(n)]
    return sum(complete(model, tok, "\n" + q[:-1]) == "=" + a
               for q, a in qs) / n


def seen_share(train_text, n=200, seed=12345):
    """評価の 200 問のうち、訓練データに同じ行がある問題の割合。"""
    lines = set(train_text.split("\n"))
    rng = random.Random(seed)              # drill_accuracy と同じ問題
    qs = [make_problem(rng, CFG["max_value"], "+-") for _ in range(n)]
    return sum(q + a in lines for q, a in qs) / n


def run(name):
    t0 = time.time()
    TMP.mkdir(exist_ok=True)
    d, steps = RUNS[name]
    CFG["model"] = GPTConfig(vocab_size=512, block_size=64, n_layer=2,
                             n_head=2, n_embd=d)
    CFG["pre_steps"] = steps
    pipeline.OUT = TMP                 # 本文の mini_pretrain.csv を消さない
    tr_text, va_text = stage_data(None)
    tok = stage_tokenizer(tr_text)
    model, _ = stage_pretrain(tok, tr_text, va_text)
    last = list(csv.DictReader(open(TMP / "mini_pretrain.csv")))[-1]
    (l_nov, n_nov), (l_dri, n_dri) = split_loss(model, tok, va_text)
    drill = drill_accuracy(model, tok, max_value=CFG["max_value"])
    eq = drill_eq(model, tok)
    seen = seen_share(tr_text)
    sft_loss = stage_sft(model, tok)
    chat = acc(model, tok)
    row = dict(name=name, n_embd=d, pre_steps=steps,
               params=model.num_params(), val_loss=float(last["val_loss"]),
               val_novel=l_nov, val_drill=l_dri,
               drill_share=n_dri / (n_nov + n_dri), drill_acc=drill,
               drill_eq=eq,
               seen=seen, sft_loss=sft_loss, chat_acc=chat,
               sec=time.time() - t0)
    with open(OUT / f"sweep_scale_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    for k, v in row.items():
        print(f"{k:12s} {v:.4f}" if isinstance(v, float) else f"{k:12s} {v}")


def combine():
    """次元を変えた 3 本と、ステップ数を変えた 3 本を、別々の CSV にする。"""
    for out, names in (("size", ["e48", "e96", "e192"]),
                       ("steps", ["s300", "e96", "s1200"])):
        rows = [next(csv.DictReader(open(OUT / f"sweep_scale_{n}.csv")))
                for n in names]
        with open(OUT / f"sweep_scale_{out}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(out, r["name"], r["params"], r["val_loss"][:5],
                  r["drill_eq"], r["chat_acc"])


if __name__ == "__main__":
    combine() if sys.argv[1] == "combine" else run(sys.argv[1])
