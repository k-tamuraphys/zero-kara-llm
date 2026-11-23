"""第 8 章: ドリルの損失を「問題の部分」と「答えの数字」に分けて測る。

実行: cd code && python ch08/drill_parts.py
      （先に mlp_lm.py、window_sweep.py 8、rnn_lm.py を実行しておく）
出力: ch08/out/drill_parts.txt
"""
import torch
import torch.nn.functional as F

import mlp_lm
import rnn_lm
from capture import Capture
from llmbook.corpus import DATA_DIR
from llmbook.train import load_checkpoint
from mlp_lm import MLPLM, VAL_NOVEL, V, tok, val

torch.set_num_threads(2)
torch.manual_seed(0)
cap = Capture()
CK = DATA_DIR.parent / "checkpoints"
EQ, NL = tok.encode("=")[0], tok.encode("\n")[0]

# --- 検証データのドリルの各トークンに、種類の印を付ける ---
# 0: 問題（行の先頭から「=」まで）  1: 答えの数字  2: 改行
# [snippet: kind]
kind = torch.zeros(len(val), dtype=torch.long)
in_answer = False
for k in range(VAL_NOVEL + 1, len(val)):
    t = int(val[k])
    if t == NL:
        kind[k], in_answer = 2, False         # 改行で 1 問が終わる
    elif in_answer:
        kind[k] = 1                           # 「=」より後ろは答えの数字
    elif t == EQ:
        in_answer = True                      # 「=」までは問題の部分
# [end: kind]
START = VAL_NOVEL + 200                       # どのモデルも測れる位置から
drill = torch.zeros(len(val), dtype=torch.bool)
drill[START:len(val) - 1] = True              # 末尾の <|endoftext|> は除く


def report(name, losses, offset):
    """losses[k] は val[k + offset] を当てる損失。種類ごとに平均する。"""
    full = torch.full((len(val),), float("nan"))
    full[offset:offset + len(losses)] = losses
    ok = drill & ~full.isnan()                # 損失を測れた位置だけ
    part = [full[ok & (kind == c)].mean().item() for c in range(3)]
    print(f"{name:<14s}{part[0]:8.3f}{part[1]:8.3f}{part[2]:8.3f}"
          f"{full[ok].mean().item():8.3f}")


share = [(drill & (kind == c)).sum().item() / drill.sum().item()
         for c in range(3)]
print(f"ドリルのトークンの内訳: 問題 {share[0]:.0%}  答えの数字 {share[1]:.0%}"
      f"  改行 {share[2]:.0%}")
print(f"{'モデル':<12s}{'問題':>6s}{'答え':>6s}{'改行':>6s}{'平均':>6s}")

for n, path in [(3, mlp_lm.CKPT), (8, CK / "ch08_mlp_n8.pt")]:
    model = MLPLM(V, n, 64, 128)
    model.load_state_dict(torch.load(path)["model"])
    report(f"MLP（n={n}）", mlp_lm.token_losses(model, val, n), n)

ck = torch.load(rnn_lm.CKPT)
model = rnn_lm.RNNLM(V, ck["d"], ck["h"])
model.load_state_dict(ck["model"])
report("RNN", rnn_lm.token_losses(model, val, ck["T"]), ck["T"])

# 基盤モデル: preview_gpt.py と同じく、長さ 128 の重ならない窓で測る
model = load_checkpoint(CK / "base.pt")
model.eval()
T = 128
n_win = (len(val) - 1) // T
x = val[:n_win * T].view(n_win, T)
y = val[1:n_win * T + 1].view(n_win, T)
with torch.no_grad():
    out = [F.cross_entropy(model(x[k:k + 32])[0].flatten(0, 1),
                           y[k:k + 32].flatten(), reduction="none")
           for k in range(0, n_win, 32)]
report("基盤モデル", torch.cat(out), 1)
cap.save("drill_parts.txt")
