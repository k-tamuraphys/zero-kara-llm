"""スイープ: 注意のヘッド数 H だけを変えて、計算ドリルの成績を比べる。

average_vs_attention.py の「注意」モデルと同じ作り・同じ学習で、
H を 1, 2, 4, 8 の 4 通りに変える（次元 D=64 は固定なので、
1 ヘッドの次元は D/H = 64, 32, 16, 8 と小さくなる。パラメータ数は同じ）。
乱数の種を 0 と 1 の 2 通りで走らせ、種による幅も見る。
1 回の実行で 1 つの設定だけを走らせ、結果を csv に 1 行ずつ足していく
（使い方: python ch09/sweep_heads.py H 種。引数なしなら全 8 通り）。
正答率は、評価用の 500 問全体と、そのうち学習データに無い問題だけの
2 通りで測る。
"""

import csv
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmbook.arith import make_problem

torch.set_num_threads(2)
OUT = Path(__file__).parent / "out"
CHARS = "0123456789+=\n"
stoi = {c: i for i, c in enumerate(CHARS)}
T, D = 12, 64  # 文脈長, 次元（average_vs_attention.py と同じ）


def make_lines(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(make_problem(rng, max_value=99, ops="+")) + "\n"
            for _ in range(n)]


def to_batch(lines: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """"\n12+35=47\n" を (B, T) にし、答えの位置だけを正解にする。"""
    xs, ys = [], []
    for line in lines:
        ids = [stoi[c] for c in "\n" + line]
        x, y = ids[:-1], ids[1:]
        eq = x.index(stoi["="])
        y = [t if i >= eq else -100 for i, t in enumerate(y)]  # 答えだけ
        pad = T - len(x)
        xs.append(x + [stoi["\n"]] * pad)
        ys.append(y + [-100] * pad)
    return torch.tensor(xs), torch.tensor(ys)


class Mixer(nn.Module):
    """縮小付き内積注意（H ヘッド）。average_vs_attention.py の注意と同じ。"""

    def __init__(self, H: int):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))

    def forward(self, x):  # x: (B, T, D)
        B, T_, _ = x.shape
        H = self.H
        q, k, v = self.qkv(x).view(B, T_, 3, H, D // H).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (B, H, T, D/H)
        m = self.mask[:T_, :T_]
        S = q @ k.transpose(-2, -1) / math.sqrt(D // H)
        A = F.softmax(S.masked_fill(m == 0, float("-inf")), dim=-1)
        y = (A @ v).transpose(1, 2).reshape(B, T_, D)
        return self.proj(y)


class TinyLM(nn.Module):
    """埋め込み → 注意 → MLP → 出力（average_vs_attention.py と同じ形）。"""

    def __init__(self, H: int):
        super().__init__()
        self.tok = nn.Embedding(len(CHARS), D)
        self.pos = nn.Embedding(T, D)
        self.mix = Mixer(H)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(),
                                 nn.Linear(4 * D, D))
        self.out = nn.Linear(D, len(CHARS))

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        x = x + self.mix(x)
        x = x + self.mlp(x)
        return self.out(x)


@torch.no_grad()
def accuracy(model, lines: list[str]) -> float:
    """"\n12+35=" から貪欲に生成し、答えが完全一致する割合。"""
    n_ok = 0
    for line in lines:
        q, a = line.split("=")
        ids = [stoi[c] for c in "\n" + q + "="]
        for _ in range(4):
            logits = model(torch.tensor([ids]))
            ids.append(logits[0, -1].argmax().item())
            if ids[-1] == stoi["\n"]:
                break
        pred = "".join(CHARS[i] for i in ids).split("=")[1].split("\n")[0]
        n_ok += pred == a.strip()
    return n_ok / len(lines)


train_lines = make_lines(20000, seed=0)
test_lines = make_lines(500, seed=1)
seen = set(train_lines)
unseen_lines = [s for s in test_lines if s not in seen]  # 学習データに無い問題
print(f"評価用 {len(test_lines)} 問のうち、学習データに無い問題: "
      f"{len(unseen_lines)} 問")
x_test, y_test = to_batch(test_lines)
configs = [(int(sys.argv[1]), int(sys.argv[2]))] if len(sys.argv) == 3 \
    else [(H, seed) for H in [1, 2, 4, 8] for seed in [0, 1]]
path = OUT / "sweep_heads.csv"
if not path.exists():
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["H", "d_head", "seed", "loss", "acc",
                                "acc_unseen"])
for H, seed in configs:
    torch.manual_seed(seed)
    model = TinyLM(H)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed)
    t0 = time.time()
    for step in range(3001):
        x, y = to_batch(rng.sample(train_lines, 128))
        loss = F.cross_entropy(model(x).view(-1, len(CHARS)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_loss = F.cross_entropy(
            model(x_test).view(-1, len(CHARS)), y_test.view(-1)).item()
    acc = accuracy(model, test_lines)
    acc_new = accuracy(model, unseen_lines)
    print(f"H={H} (d_head={D // H:2d}) 種 {seed}: 答えの損失 "
          f"{test_loss:.3f}  正答率 {acc:6.1%}  "
          f"学習データに無い問題 {acc_new:6.1%}  "
          f"({time.time() - t0:.0f} 秒)", flush=True)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([H, D // H, seed, f"{test_loss:.4f}",
                                f"{100 * acc:.1f}", f"{100 * acc_new:.1f}"])
