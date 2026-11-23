"""第 8 章: 学習した RNN を調べる（読む長さと損失、昔のトークンへの勾配）。

実行: cd code && python ch08/rnn_probe.py   （先に rnn_lm.py を実行しておく）
出力: ch08/out/rnn_context.csv, rnn_context.txt, rnn_grad.csv, rnn_grad.txt
"""
import torch
import torch.nn.functional as F

from capture import OUT, Capture
from mlp_lm import VAL_NOVEL, V, split_mean, val
from rnn_lm import CKPT, RNNLM, token_losses

torch.set_num_threads(2)
torch.manual_seed(0)
cap = Capture()
ck = torch.load(CKPT)
model = RNNLM(V, ck["d"], ck["h"])
model.load_state_dict(ck["model"])

# --- 1. 同じモデルに、直前の k 個だけを読ませて次を当てさせる ---
print("読む個数   全体   小説  ドリル")
with open(OUT / "rnn_context.csv", "w") as f:
    # d_novel, d_drill は「1 個だけ読んだとき」からの損失の変化（図に使う）
    f.write("k,val,val_novel,val_drill,d_novel,d_drill\n")
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        a, b, c = split_mean(token_losses(model, val, k), k)
        if k == 1:
            b1, c1 = b, c
        f.write(f"{k},{a:.4f},{b:.4f},{c:.4f},{b - b1:.4f},{c - c1:.4f}\n")
        print(f"{k:6d} {a:8.3f}{b:7.3f}{c:7.3f}")
cap.save("rnn_context.txt")


# --- 2. 最後の予測の損失は、何個前のトークンまで勾配を届けるか ---
def grad_by_distance(start, stop, T=32, B=512):
    """val[start:stop] から B 本の列を取り、位置ごとの勾配の大きさを返す。"""
    i = torch.randint(start, stop - T, (B,))
    x = val[i[:, None] + torch.arange(T)]         # (B, T)
    y = val[i + T]                                # (B,) 列の次のトークン
    # [snippet: grad]
    e = model.emb(x).detach().requires_grad_(True)   # (B, T, d)
    h = torch.zeros(B, model.h_dim)
    for t in range(T):
        h = torch.tanh(model.fx(e[:, t]) + model.fh(h))
    loss = F.cross_entropy(model.out(h), y)       # 最後の予測の損失だけ
    loss.backward()
    g = e.grad.norm(dim=2).mean(dim=0)            # (T,) 位置ごとの大きさ
    # [end: grad]
    return (g / g[-1]).flip(0)                    # 1 個前を 1 として、近い順に


g_novel = grad_by_distance(0, VAL_NOVEL)
g_drill = grad_by_distance(VAL_NOVEL + 1, len(val) - 1)
with open(OUT / "rnn_grad.csv", "w") as f:
    f.write("dist,novel,drill\n")
    for k in range(len(g_novel)):
        f.write(f"{k + 1},{g_novel[k]:.3e},{g_drill[k]:.3e}\n")
print("何個前   小説     ドリル   （1 個前の勾配の大きさを 1 とする）")
for k in [1, 2, 4, 8, 16, 32]:
    print(f"{k:5d}  {g_novel[k - 1]:.1e}  {g_drill[k - 1]:.1e}")
cap.save("rnn_grad.txt")
