"""第 8 章: 窓の幅 n を変えて MLP 言語モデルを学習し、固定窓の限界を測る。

実行: cd code && python ch08/window_sweep.py 6   （窓の幅を 1 つ渡す。2〜3 分）
      n = 1, 2, 4, 6, 8, 12 について 1 回ずつ実行する。
      n = 3 は mlp_lm.py が保存した学習済みモデルを読み込む（学習し直さない）。
      引数なしで実行すると、そろった結果を表にまとめるだけ。
出力: ch08/out/window_n<n>.csv（1 行ずつ）→ ch08/out/window.csv, window.txt
      n = 8 のときだけ、モデルを checkpoints/ch08_mlp_n8.pt に保存する。
"""
import random
import sys
import time

import torch

from capture import OUT
from llmbook.arith import make_problem
from mlp_lm import (CKPT, MLPLM, V, drill_accuracy, mlp_next_token,
                    split_mean, tok, token_losses, train_model, val)

torch.set_num_threads(2)
D, H, STEPS = 64, 128, 3000                 # mlp_lm.py と同じ設定


def fits_in_window(n, n_problems=200, seed=12345):
    """答えの 1 文字目を当てる時点で、問題文の全体が窓に入っている割合。"""
    rng = random.Random(seed)
    qs = [make_problem(rng)[0] for _ in range(n_problems)]
    return sum(len(tok.encode(q)) <= n for q in qs) / n_problems


def run(n):
    torch.manual_seed(0)
    t0 = time.time()
    # [snippet: sweep]
    model = MLPLM(V, n, D, H)               # 変えるのは窓の幅 n だけ
    if n == 3:
        model.load_state_dict(torch.load(CKPT)["model"])
    else:
        train_model(model, n, STEPS)
    if n == 8:                              # drill_parts.py で調べるので保存
        torch.save({"model": model.state_dict(), "n": n, "d": D, "h": H},
                   CKPT.with_name("ch08_mlp_n8.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    loss = split_mean(token_losses(model, val, n), n)   # 全体, 小説, ドリル
    acc = drill_accuracy(mlp_next_token(model, n))
    # [end: sweep]
    row = (f"{n},{n_params},{loss[0]:.4f},{loss[1]:.4f},{loss[2]:.4f},"
           f"{100 * acc:.1f},{100 * fits_in_window(n):.1f},"
           f"{time.time() - t0:.0f}")
    (OUT / f"window_n{n}.csv").write_text(row + "\n")
    print(row)


def collect():
    rows = []
    for path in OUT.glob("window_n*.csv"):
        rows.append(path.read_text().strip().split(","))
    rows.sort(key=lambda r: int(r[0]))
    head = "n,params,val,val_novel,val_drill,acc,fit,sec"
    (OUT / "window.csv").write_text(
        head + "\n" + "".join(",".join(r) + "\n" for r in rows))
    lines = ["窓の幅 パラメータ数   全体   小説  ドリル  正答率  窓に入る"]
    for n, p, a, b, c, acc, fit, _ in rows:
        lines.append(f"{int(n):5d} {int(p):11,} {float(a):7.3f}"
                     f"{float(b):7.3f}{float(c):7.3f} {float(acc):6.1f}%"
                     f" {float(fit):7.1f}%")
    (OUT / "window.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(int(arg))
    collect()
