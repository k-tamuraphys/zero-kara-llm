"""第 5 章（図のための追加）: ドロップアウトを 3 回続けてかけたときの出力。

regularize.py の dropout と同じ計算を、同じ乱数の種 0 から 3 回続けて行う。
1 回目は regularize.py の実行結果（本文の (0,2,0,0,0,2,0,2)）と同じになる。
（regularize.py を読み込むと過学習の実験も走るので、関数を写してある。）
    cd code && python ch05/fig_dropout_masks.py \
        > ch05/out/fig_dropout_masks.txt
"""
import torch


def dropout(h, p, training):
    """学習中だけ、各成分を確率 p で 0 にし、残りを 1/(1-p) 倍する。"""
    if not training or p == 0.0:
        return h
    keep = (torch.rand_like(h) >= p).float()  # 残す成分が 1、消す成分が 0
    return h * keep / (1.0 - p)


torch.manual_seed(0)
h = torch.ones(2, 8)                          # regularize.py と同じ形
for k in range(1, 4):
    out = dropout(h, 0.5, True)[0]
    zeros = int((out == 0).sum())
    print(f"{k} 回目の順伝播: {[int(v) for v in out.tolist()]}"
          f"（0 が {zeros} 個、平均 {out.mean():.2f}）")
