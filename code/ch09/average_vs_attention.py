"""実験: 「平均」と「注意」で、計算ドリルの答えを当てる力を比べる。

3 つのモデルは、文脈の混ぜ方（重み行列 A の決め方）だけが違う。
  none:      混ぜない（自分のトークンしか見ない）
  average:   過去の平均（A は固定の下三角平均行列）
  attention: 縮小付き内積注意（A を内容から計算する、4 ヘッド）
"""

import csv
import math
import random
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
T, D, H = 12, 64, 4  # 文脈長, 次元, ヘッド数


def make_lines(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(make_problem(rng, max_value=99, ops="+")) + "\n"
            for _ in range(n)]


def to_batch(lines: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """"\n12+35=47\n" を右詰めの (B, T) にし、答えの位置だけを正解にする。"""
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


# [snippet: mixer]
class Mixer(nn.Module):
    """文脈を混ぜる層。混ぜ方 kind だけが違い、ほかは同じ。"""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))

    def forward(self, x):  # x: (B, T, D)
        B, T_, _ = x.shape
        q, k, v = self.qkv(x).view(B, T_, 3, H, D // H).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (B, H, T, D/H)
        m = self.mask[:T_, :T_]
        if self.kind == "none":  # 自分だけ: A は単位行列
            A = torch.eye(T_).expand(B, H, T_, T_)
        elif self.kind == "average":  # 過去の平均: A は固定
            A = (m / m.sum(dim=1, keepdim=True)).expand(B, H, T_, T_)
        else:  # 注意: A を q と k から計算する
            S = q @ k.transpose(-2, -1) / math.sqrt(D // H)
            A = F.softmax(S.masked_fill(m == 0, float("-inf")), dim=-1)
        y = (A @ v).transpose(1, 2).reshape(B, T_, D)
        return self.proj(y)
# [end: mixer]


class TinyLM(nn.Module):
    """埋め込み → 混ぜる層 → MLP → 出力。混ぜる層の後に残差を足す。"""

    def __init__(self, kind: str):
        super().__init__()
        self.tok = nn.Embedding(len(CHARS), D)
        self.pos = nn.Embedding(T, D)  # 位置ごとのベクトル（第 10 章）
        self.mix = Mixer(kind)
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
x_test, y_test = to_batch(test_lines)
log_rows, results = [], []
for kind in ["none", "average", "attention"]:
    torch.manual_seed(0)
    model = TinyLM(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(0)
    t0 = time.time()
    for step in range(3001):
        x, y = to_batch(rng.sample(train_lines, 128))
        loss = F.cross_entropy(model(x).view(-1, len(CHARS)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                test_loss = F.cross_entropy(
                    model(x_test).view(-1, len(CHARS)), y_test.view(-1))
            log_rows.append((kind, step, loss.item(), test_loss.item()))
    acc = accuracy(model, test_lines)
    results.append((kind, test_loss.item(), acc, time.time() - t0))
    print(f"{kind:>9}: 答えの損失 {test_loss.item():.3f}  "
          f"正答率 {acc:6.1%}  ({time.time() - t0:.0f} 秒)")
    if kind == "attention":
        torch.save(model.state_dict(),
                   OUT.parent.parent / "checkpoints" / "ch09_tiny_attn.pt")

with open(OUT / "average_vs_attention.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "none", "average", "attention"])
    steps = sorted({r[1] for r in log_rows})
    for s in steps:
        w.writerow([s] + [f"{r[3]:.4f}" for k in ["none", "average",
                    "attention"] for r in log_rows if r[0] == k and r[1] == s])
