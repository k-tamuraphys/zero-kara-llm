"""「混ぜない」「平均」のモデルを、学習データに無い問題だけでも採点する。

average_vs_attention.py と同じモデル・同じ学習（乱数の種 0）を走らせ直し、
評価用の 500 問全体と、そのうち学習データに無い問題だけの正答率を出す。
（「注意」の 4 ヘッド・種 0 は sweep_heads.py の H=4、種 0 と同じ設定。）
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmbook.arith import make_problem

torch.set_num_threads(2)
CHARS = "0123456789+=\n"
stoi = {c: i for i, c in enumerate(CHARS)}
T, D, H = 12, 64, 4  # 文脈長, 次元, ヘッド数


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
    """文脈を混ぜる層（average_vs_attention.py と同じ）。"""

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
        if self.kind == "none":  # 自分だけ
            A = torch.eye(T_).expand(B, H, T_, T_)
        elif self.kind == "average":  # 過去の平均
            A = (m / m.sum(dim=1, keepdim=True)).expand(B, H, T_, T_)
        else:  # 注意
            S = q @ k.transpose(-2, -1) / math.sqrt(D // H)
            A = F.softmax(S.masked_fill(m == 0, float("-inf")), dim=-1)
        y = (A @ v).transpose(1, 2).reshape(B, T_, D)
        return self.proj(y)


class TinyLM(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.tok = nn.Embedding(len(CHARS), D)
        self.pos = nn.Embedding(T, D)
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
seen = set(train_lines)
unseen_lines = [s for s in test_lines if s not in seen]
print(f"評価用 {len(test_lines)} 問のうち、学習データに無い問題: "
      f"{len(unseen_lines)} 問")
for kind in ["none", "average"]:
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
    acc = accuracy(model, test_lines)
    acc_new = accuracy(model, unseen_lines)
    print(f"{kind:>9}: 正答率 {acc:6.1%}  学習データに無い問題 "
          f"{acc_new:6.1%}  ({time.time() - t0:.0f} 秒)")
