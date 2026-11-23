"""第 8 章: 基盤モデル base.pt を同じ検証データで測る（行き先の先取り）。

モデルの中身は第 9〜10 章、学習は第 13 章で扱う。ここでは読むだけ。
実行: cd code && python ch08/preview_gpt.py
出力: ch08/out/preview_gpt.txt
"""
import torch
import torch.nn.functional as F

from capture import Capture
from llmbook.arith import drill_accuracy
from llmbook.corpus import DATA_DIR
from llmbook.train import load_checkpoint
from mlp_lm import split_mean, tok, val

torch.set_num_threads(2)
torch.manual_seed(0)
cap = Capture()
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()

T = 128                                   # 基盤モデルが一度に見られる長さ
n_win = (len(val) - 1) // T
x = val[:n_win * T].view(n_win, T)        # (窓の数, T)
y = val[1:n_win * T + 1].view(n_win, T)   # 1 つずらした正解
losses = []
with torch.no_grad():
    for i in range(0, n_win, 32):
        logits, _ = model(x[i:i + 32])    # (B, T, V)
        losses.append(F.cross_entropy(logits.flatten(0, 1),
                                      y[i:i + 32].flatten(),
                                      reduction="none"))
a, b, c = split_mean(torch.cat(losses), 1)
acc = drill_accuracy(model, tok)
print(f"パラメータ数 {model.num_params():,}")
print(f"基盤モデル（直前 {T} 個まで）  全体 {a:.3f}  小説 {b:.3f}"
      f"  ドリル {c:.3f}  正答率 {acc:.1%}")
cap.save("preview_gpt.txt")
