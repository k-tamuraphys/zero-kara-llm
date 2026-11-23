"""第 8 章: 固定窓の MLP 言語モデル（Bengio 2003 型）を BPE トークンで学習。

実行: cd code && python ch08/mlp_lm.py
出力: ch08/out/mlp_*.txt, ch08/out/mlp_log.csv, checkpoints/ch08_mlp_lm.pt
"""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmbook.arith import make_problem
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import lr_at

torch.set_num_threads(2)
OUT = Path(__file__).parent / "out"
CKPT = DATA_DIR.parent / "checkpoints" / "ch08_mlp_lm.pt"
V = 4096                                   # 語彙サイズ


def load_tokens(name):
    """トークン番号の列（uint16 で保存されている）を読み込む。"""
    a = np.fromfile(DATA_DIR / name, dtype=np.uint16)
    return torch.from_numpy(a.astype(np.int64))


tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
train, val = load_tokens("train.bin"), load_tokens("val.bin")
EOT = tok.special_tokens["<|endoftext|>"]
# 検証データは「小説 <|endoftext|> ドリル <|endoftext|>」の順に並んでいる
VAL_NOVEL = int((val == EOT).nonzero()[0])  # 小説の部分が終わる位置


# [snippet: batch]
def get_batch(data, B, n):
    """data からランダムに B か所を選び、n 個の窓と、その次の 1 個を返す。"""
    i = torch.randint(len(data) - n, (B,))  # 窓の先頭の位置
    x = data[i[:, None] + torch.arange(n)]  # (B, n) 直前の n 個
    y = data[i + n]                         # (B,)   次の 1 個（正解）
    return x, y
# [end: batch]


# [snippet: model]
class MLPLM(nn.Module):
    """直前 n 個のトークンから、次のトークンのロジットを出す。"""

    def __init__(self, V, n, d, h):
        super().__init__()
        self.emb = nn.Embedding(V, d)       # 埋め込み表 E: (V, d)
        self.fc1 = nn.Linear(n * d, h)      # 並べたベクトル → 隠れ層
        self.fc2 = nn.Linear(h, V)          # 隠れ層 → V 個のロジット

    def forward(self, x):                   # x: (B, n) トークン番号
        e = self.emb(x)                     # (B, n, d) 表を引く
        e = e.view(e.shape[0], -1)          # (B, n*d)  横に並べて 1 本に
        h = torch.tanh(self.fc1(e))         # (B, h)
        return self.fc2(h)                  # (B, V)    ロジット
# [end: model]


@torch.no_grad()
def token_losses(model, data, n):
    """data の n 番目以降の各トークンについて、損失 -log p を返す。"""
    x = data.unfold(0, n, 1)[:-1]           # (N-n, n) すべての窓
    y = data[n:]                            # (N-n,)
    out = [F.cross_entropy(model(x[k:k + 8192]), y[k:k + 8192],
                           reduction="none") for k in range(0, len(y), 8192)]
    return torch.cat(out)


def split_mean(losses, offset):
    """検証データの損失を（全体, 小説, ドリル）に分けて平均する。

    losses[k] は val[k + offset] を当てる損失。
    """
    cut = VAL_NOVEL - offset
    return (losses.mean().item(), losses[:cut + 1].mean().item(),
            losses[cut + 1:].mean().item())


# [snippet: train]
def train_model(model, n, steps, B=1024, lr=1.5e-2, n_eval=8):
    """標準の学習ループ。途中の損失を (step, 訓練, 検証…) の表で返す。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log, recent = [], []
    for step in range(1, steps + 1):
        for g in opt.param_groups:          # ウォームアップ + コサイン減衰
            g["lr"] = lr_at(step, lr, 100, steps)
        x, y = get_batch(train, B, n)
        loss = F.cross_entropy(model(x), y)  # 予測して、損失を測る
        opt.zero_grad()
        loss.backward()                     # 勾配を求める
        opt.step()                          # パラメータを更新する
        recent.append(loss.item())
        if step % (steps // n_eval) == 0:
            va = split_mean(token_losses(model, val, n), n)
            log.append((step, sum(recent) / len(recent), *va))
            print(f"step {step:5d}  訓練 {log[-1][1]:.3f}  検証 {va[0]:.3f}"
                  f"（小説 {va[1]:.3f}  ドリル {va[2]:.3f}）")
            recent = []
    return log
# [end: train]


# [snippet: baseline]
def bigram_count_losses(alpha):
    """数えるだけのバイグラム（第 6 章）を BPE トークンで作り、検証する。"""
    C = torch.bincount(train[:-1] * V + train[1:], minlength=V * V)
    C = C.view(V, V).double()               # (V, V) 組の出現回数
    P = (C + alpha) / (C.sum(dim=1, keepdim=True) + alpha * V)
    return -torch.log(P[val[:-1], val[1:]])
# [end: baseline]


def unigram_losses(alpha=0.1):
    c = torch.bincount(train, minlength=V).double()
    p = (c + alpha) / (c.sum() + alpha * V)
    return -torch.log(p[val])


# [snippet: generate]
@torch.no_grad()
def generate(model, n, ids, max_new, temperature=1.0):
    """ids の続きを max_new 個、1 個ずつ書き足す。見るのは末尾の n 個だけ。"""
    ids = list(ids)
    for _ in range(max_new):
        logits = model(torch.tensor([ids[-n:]]))[0]        # (V,)
        if temperature == 0:
            ids.append(int(logits.argmax()))               # 貪欲
        else:
            p = torch.softmax(logits / temperature, dim=0)
            ids.append(int(torch.multinomial(p, 1)))
    return ids
# [end: generate]


DRILL_PREFIX = "12+35=47\n"                 # 問題の前に置く、ドリルの 1 行


def drill_accuracy(next_token, n_problems=200, seed=12345):
    """計算ドリルの正答率。next_token(ids) は続きの 1 トークンを返す関数。

    問題は llmbook.arith.drill_accuracy と同じ 200 問（同じ種）。
    """
    rng = random.Random(seed)
    newline = tok.encode("\n")[0]
    n_correct = 0
    for _ in range(n_problems):
        question, answer = make_problem(rng)
        ids, out = tok.encode(DRILL_PREFIX + question), []
        for _ in range(8):
            t = next_token(ids + out)
            if t == newline:
                break
            out.append(t)
        n_correct += tok.decode(out) == answer
    return n_correct / n_problems


def mlp_next_token(model, n):
    return lambda ids: int(model(torch.tensor([ids[-n:]])).argmax())


def main():
    from capture import Capture
    cap = Capture()
    torch.manual_seed(0)
    n, d, h = 3, 64, 128                    # 窓の幅、埋め込みの次元、隠れ層

    # --- データ: 窓と正解の例 ---
    print(f"訓練 {len(train):,} トークン  検証 {len(val):,} トークン"
          f"（うち小説 {VAL_NOVEL:,}）")
    ids = tok.encode("私はその人を常に先生と呼んでいた。")
    print("トークン列:", " ".join(f"「{tok.token_str(t)}」" for t in ids))
    for k in [0, 1, 3]:
        x, y = ids[k:k + n], ids[k + n]
        ctx = "".join(f"「{tok.token_str(t)}」" for t in x)
        print(f"入力 {str(x):18s} {ctx} → 正解 {y:4d}「{tok.token_str(y)}」")
    cap.save("mlp_data.txt")

    # --- モデル ---
    # [snippet: build]
    model = MLPLM(V, n, d, h)
    for name, p in model.named_parameters():
        print(f"{name:11s} {str(tuple(p.shape)):12s} {p.numel():8,}")
    print(f"合計 {sum(p.numel() for p in model.parameters()):,}")
    x, y = get_batch(train, 4, n)
    logits = model(x)
    print("入力", tuple(x.shape), "→ ロジット", tuple(logits.shape))
    print(f"学習前の損失 {F.cross_entropy(logits, y).item():.3f}"
          f"  (log V = {math.log(V):.3f})")
    # [end: build]
    cap.save("mlp_build.txt")

    # --- 学習 ---
    steps = 3000
    t0 = time.time()
    log = train_model(model, n, steps)
    train_sec = time.time() - t0
    cap.save("mlp_train.txt")
    with open(OUT / "mlp_log.csv", "w") as f:
        f.write("step,train,val,val_novel,val_drill\n")
        for row in log:
            f.write(f"{row[0]}," + ",".join(f"{v:.4f}" for v in row[1:])
                    + "\n")
    torch.save({"model": model.state_dict(), "n": n, "d": d, "h": h}, CKPT)

    # --- 基準線との比較（検証データ、1 トークンあたりの損失）---
    rows = [("一様（log V）", (math.log(V),) * 3),
            ("ユニグラム", split_mean(unigram_losses(), 0))]
    best = min([0.003, 0.01, 0.03, 0.1, 0.3],
               key=lambda a: bigram_count_losses(a).mean().item())
    rows.append((f"バイグラム（数える, α={best}）",
                 split_mean(bigram_count_losses(best), 1)))
    rows.append((f"MLP（直前 {n} 個）",
                 split_mean(token_losses(model, val, n), n)))
    print(f"{'モデル':<22s}{'全体':>7s}{'小説':>7s}{'ドリル':>7s}")
    for name, (a, b, c) in rows:
        print(f"{name:<22s}{a:8.3f}{b:8.3f}{c:8.3f}")
    cap.save("mlp_compare.txt")

    # --- 文章の生成 ---
    torch.manual_seed(0)
    for prompt in ["私はその人を常に", "\n25+48="]:
        ids = generate(model, n, tok.encode(prompt), 30, temperature=0.8)
        print(repr(tok.decode(ids)))
    cap.save("mlp_generate.txt")

    # --- 計算ドリル ---
    acc = drill_accuracy(mlp_next_token(model, n))
    print(f"ドリルの正答率 {acc:.1%}")
    cap.save("mlp_drill.txt")
    print(f"学習時間 {train_sec:.0f} 秒")
    cap.save("mlp_time.txt")


if __name__ == "__main__":
    main()
