"""第 8 章: RNN を学習させる列の長さ T を変えると、使える文脈はどう変わるか。

実行: cd code && python ch08/sweep_rnn_len.py 4   （T を 1 つ渡す。5〜7 分）
      T = 4, 8 について 1 回ずつ実行する。T = 16 は rnn_lm.py が
      保存した学習済みモデルを読み込む（学習し直さない）。
      T = 32 は、共有の計算機で 15 分を超えたので外した（図は T=4,8,16）。
      引数なしで実行すると、そろった結果を表にまとめるだけ。
設定は rnn_lm.py と同じ（d=64, d_h=128, 2000 回, 学習率 0.008）。
1 回の更新で当てる個数を 1024 にそろえるため、列の本数を B = 1024 / T にする。
出力: ch08/out/rnn_len_T<T>.csv → ch08/out/rnn_len.csv, rnn_len.txt
"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from capture import OUT
from llmbook.train import lr_at
from mlp_lm import V, split_mean, train, val
from rnn_lm import CKPT, RNNLM, get_seq_batch, token_losses

torch.set_num_threads(2)
D, H, STEPS, LR = 64, 128, 2000, 8e-3
KS = [1, 2, 4, 8, 16, 32, 64]                # 検証で読ませる個数


def run(T):
    torch.manual_seed(0)
    t0 = time.time()
    model = RNNLM(V, D, H)
    if T == 16:                              # rnn_lm.py で学習したもの
        model.load_state_dict(torch.load(CKPT)["model"])
    else:
        B = 1024 // T
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        for step in range(1, STEPS + 1):     # rnn_lm.py と同じ学習ループ
            for g in opt.param_groups:
                g["lr"] = lr_at(step, LR, 100, STEPS)
            x, y = get_seq_batch(train, B, T)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    sec = time.time() - t0
    lines = []
    for k in KS:                             # 直前の k 個だけを読ませる
        a, b, c = split_mean(token_losses(model, val, k), k)
        lines.append(f"{T},{k},{a:.4f},{b:.4f},{c:.4f},{sec:.0f}\n")
        print(lines[-1], end="")
    (OUT / f"rnn_len_T{T}.csv").write_text("".join(lines))


def collect():
    rows = []
    for path in sorted(OUT.glob("rnn_len_T*.csv")):
        rows += [r.split(",") for r in path.read_text().split()]
    rows.sort(key=lambda r: (int(r[0]), int(r[1])))
    head = "T,k,val,val_novel,val_drill,sec\n"
    (OUT / "rnn_len.csv").write_text(
        head + "".join(",".join(r) + "\n" for r in rows))
    lines = ["学習の長さ T  読む個数 k   全体   小説  ドリル"]
    for T, k, a, b, c, _ in rows:
        lines.append(f"{int(T):10d} {int(k):10d} {float(a):7.3f}"
                     f"{float(b):7.3f}{float(c):7.3f}")
    (OUT / "rnn_len.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(int(arg))
    collect()
