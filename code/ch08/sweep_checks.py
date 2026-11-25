"""第 8 章: 本文の説明を数で確かめる、小さな測定をまとめたもの。

実行: cd code && python ch08/sweep_checks.py > ch08/out/sweep_checks.txt
      （先に mlp_lm.py と window_sweep.py 8 を実行しておく。1 分ほど）
出力: ch08/out/sweep_checks.txt, ch08/out/sim_hist.csv, ch08/out/fit_last.csv
既存のコードと学習済みモデルは読むだけで、何も書き換えない。
"""
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from capture import OUT
from llmbook.arith import make_problem
from mlp_lm import CKPT, EOT, MLPLM, V, tok, train

torch.set_num_threads(2)
torch.manual_seed(0)

# --- 1. 数える表は、どれだけ埋まっているか ---
pairs = torch.unique(train[:-1] * V + train[1:])
print(f"バイグラムの表 {V * V:,} マスのうち、1 回以上現れた組"
      f" {len(pairs):,} 個（{len(pairs) / (V * V):.2%}）")
seq = train.tolist()
for m in [2, 3, 4]:                      # m 個組（直前 m-1 個 + 次の 1 個）
    c = Counter(tuple(seq[i:i + m]) for i in range(len(seq) - m + 1))
    once = sum(1 for v in c.values() if v == 1)
    print(f"{m} 個組: 現れた種類 {len(c):,}  そのうち 1 回だけ"
          f" {once / len(c):.1%}")

# --- 2. ミニバッチの乱数で、一度も選ばれない窓の割合 ---
N, draws = len(train) - 3, 1024 * 3000   # mlp_lm.py と同じ（n=3）
print(f"延べ {draws:,} 回選んで、一度も選ばれない窓の割合"
      f" {math.exp(-draws / N):.1%}（(1-1/N)^回数 の近似）")

# --- 3. 答えを書き進めると、問題文は窓に入っているか ---
rng = random.Random(12345)               # 正答率と同じ 200 問
probs = [make_problem(rng) for _ in range(200)]
need = [len(tok.encode(q)) + len(a) - 1 for q, a in probs]
with open(OUT / "fit_last.csv", "w") as f:
    f.write("n,first,last\n")
    for n in [1, 2, 3, 4, 6, 8, 10, 11, 12]:
        first = sum(len(tok.encode(q)) <= n for q, _ in probs) / 200
        last = sum(k <= n for k in need) / 200
        f.write(f"{n},{100 * first:.1f},{100 * last:.1f}\n")
        print(f"n={n:2d}: 答えの 1 文字目で入る {first:.1%}"
              f"  最後の桁まで入る {last:.1%}")


# --- 4. 窓を広げたモデルは、訓練データにも合っていないか ---
@torch.no_grad()
def mean_loss(model, data, n):
    x = data.unfold(0, n, 1)[:-1]
    y = data[n:]
    s = sum(F.cross_entropy(model(x[k:k + 8192]), y[k:k + 8192],
                            reduction="sum").item()
            for k in range(0, len(y), 8192))
    return s / len(y)


novel_end = int((train == EOT).nonzero()[0])   # 訓練データの小説の終わり
head = train[:200_000]                         # 小説の先頭 20 万トークン
tail = train[novel_end - 200_000:novel_end]    # 小説の末尾 20 万トークン
for name, n in [("ch08_mlp_lm.pt", 3), ("ch08_mlp_n8.pt", 8)]:
    model = MLPLM(V, n, 64, 128)
    model.load_state_dict(torch.load(CKPT.with_name(name))["model"])
    lh, lt = mean_loss(model, head, n), mean_loss(model, tail, n)
    print(f"n={n}: 訓練データの小説での損失 先頭 {lh:.3f}  末尾 {lt:.3f}")

# --- 5. 決めた 2 個のトークンの類似度は、乱数の表ではどのくらいか ---
count = torch.bincount(train, minlength=V)
skip = (count < 50) | torch.tensor(
    ["\ufffd" in tok.token_str(i) for i in range(V)])
keep = (~skip).nonzero().flatten()           # 観察する 3,068 個
E1 = F.normalize(torch.load(CKPT)["model"]["emb.weight"], dim=1)
E0 = F.normalize(nn.Embedding(V, 64).weight.detach(), dim=1)
a = keep[torch.randint(len(keep), (20000,))]   # 無作為に選んだ 2 万組
b = keep[torch.randint(len(keep), (20000,))]
a, b = a[a != b], b[a != b]
edges = torch.linspace(-0.6, 1.0, 33)          # 幅 0.05 の区間
rows = []
for name, E in [("学習前", E0), ("学習後", E1)]:
    s = (E[a] * E[b]).sum(dim=1)
    print(f"{name}の表: 無作為な 2 個の類似度 平均 {s.mean():+.3f}"
          f"  標準偏差 {s.std():.3f}  上位 1% の境 {s.quantile(0.99):.3f}")
    for v in [0.27, 0.52, 0.77]:
        print(f"  {v:.2f} 以上になる組の割合 {(s >= v).float().mean():.2%}")
    rows.append(torch.histc(s, bins=32, min=-0.6, max=1.0) / len(s))
with open(OUT / "sim_hist.csv", "w") as f:
    f.write("x,before,after\n")                # x は区間の左端
    for k in range(32):
        f.write(f"{edges[k]:.3f},{rows[0][k]:.4f},{rows[1][k]:.4f}\n")
    f.write(f"{edges[32]:.3f},0,0\n")
