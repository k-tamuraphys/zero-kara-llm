"""第 8 章: 最小の RNN 言語モデル。隠れ状態に書き足しながら 1 個ずつ読む。

実行: cd code && python ch08/rnn_lm.py   （先に mlp_lm.py を実行しておく）
出力: ch08/out/rnn_*.txt, ch08/out/rnn_log.csv, checkpoints/ch08_rnn_lm.pt
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR
from llmbook.train import lr_at
from mlp_lm import V, drill_accuracy, split_mean, tok, train, val

torch.set_num_threads(2)
CKPT = DATA_DIR.parent / "checkpoints" / "ch08_rnn_lm.pt"


# [snippet: model]
class RNNLM(nn.Module):
    """トークンを左から 1 個ずつ読み、隠れ状態 h を更新していく言語モデル。"""

    def __init__(self, V, d, h):
        super().__init__()
        self.emb = nn.Embedding(V, d)             # 埋め込み表 E: (V, d)
        self.fx = nn.Linear(d, h)                 # W_x と b
        self.fh = nn.Linear(h, h, bias=False)     # W_h
        self.out = nn.Linear(h, V)                # 隠れ状態 → ロジット
        self.h_dim = h

    def read(self, x):                            # x: (B, T) トークン番号
        e = self.emb(x)                           # (B, T, d)
        h = torch.zeros(x.shape[0], self.h_dim)   # (B, h) 最初は空っぽ
        hs = []
        for t in range(x.shape[1]):               # 左から順に、T 回
            h = torch.tanh(self.fx(e[:, t]) + self.fh(h))   # 書き足す
            hs.append(h)
        return torch.stack(hs, dim=1)             # (B, T, h) 各時刻の h

    def forward(self, x):
        return self.out(self.read(x))             # (B, T, V) 各時刻の予測
# [end: model]

    def last_logits(self, x):
        """x を読み終えた時点の予測だけを返す。(B, T) → (B, V)"""
        return self.out(self.read(x)[:, -1])


# [snippet: batch]
def get_seq_batch(data, B, T):
    """長さ T の列と、それを 1 つずらした正解の列を B 本返す。"""
    i = torch.randint(len(data) - T, (B,))
    x = data[i[:, None] + torch.arange(T)]        # (B, T)
    y = data[i[:, None] + torch.arange(T) + 1]    # (B, T) 各位置の「次」
    return x, y
# [end: batch]


@torch.no_grad()
def token_losses(model, data, T):
    """各トークンを、直前の T 個だけを読ませて当てたときの損失。"""
    x = data.unfold(0, T, 1)[:-1]                 # (N-T, T)
    y = data[T:]
    out = [F.cross_entropy(model.last_logits(x[k:k + 4096]), y[k:k + 4096],
                           reduction="none") for k in range(0, len(y), 4096)]
    return torch.cat(out)


def rnn_next_token(model, T):
    return lambda ids: int(model.last_logits(torch.tensor([ids[-T:]]))
                           .argmax())


def main():
    from capture import OUT, Capture
    cap = Capture()
    torch.manual_seed(0)
    d, h, T, B, steps, lr = 64, 128, 16, 64, 2000, 8e-3

    model = RNNLM(V, d, h)
    for name, p in model.named_parameters():
        print(f"{name:11s} {str(tuple(p.shape)):12s} {p.numel():8,}")
    print(f"合計 {sum(p.numel() for p in model.parameters()):,}")
    x, y = get_seq_batch(train, B, T)
    print("入力", tuple(x.shape), "→ ロジット", tuple(model(x).shape))
    cap.save("rnn_build.txt")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log, recent, t0 = [], [], time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr, 100, steps)
        # [snippet: train]
        x, y = get_seq_batch(train, B, T)         # (B, T) ずつ
        logits = model(x)                         # (B, T, V)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        opt.zero_grad()
        loss.backward()                           # 時間をさかのぼって勾配
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # [end: train]
        recent.append(loss.item())
        if step % (steps // 5) == 0:
            va = split_mean(token_losses(model, val, T), T)
            log.append((step, sum(recent) / len(recent), *va))
            print(f"step {step:5d}  訓練 {log[-1][1]:.3f}  検証 {va[0]:.3f}"
                  f"（小説 {va[1]:.3f}  ドリル {va[2]:.3f}）")
            recent = []
    train_sec = time.time() - t0
    cap.save("rnn_train.txt")
    with open(OUT / "rnn_log.csv", "w") as f:
        f.write("step,train,val,val_novel,val_drill\n")
        for row in log:
            f.write(f"{row[0]}," + ",".join(f"{v:.4f}" for v in row[1:])
                    + "\n")
    torch.save({"model": model.state_dict(), "d": d, "h": h, "T": T}, CKPT)

    a, b, c = log[-1][2:]
    acc = drill_accuracy(rnn_next_token(model, T))
    print(f"RNN（直前 {T} 個まで）  全体 {a:.3f}  小説 {b:.3f}  ドリル {c:.3f}"
          f"  正答率 {acc:.1%}")
    cap.save("rnn_compare.txt")
    print(f"学習時間 {train_sec:.0f} 秒")
    cap.save("rnn_time.txt")


if __name__ == "__main__":
    main()
