"""第 6 章: 基盤モデル base.pt の、検証用の文章での 1 文字あたりの損失を測る。

n-グラムとの比較のための先取り（モデルの中身は第 8〜10 章、学習は第 13 章）。
実行: cd code && python ch06/preview_gpt.py   （base.pt は読むだけ）
"""
import math

import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
torch.manual_seed(0)
train_text, val_text = train_val_split(load_corpus())
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()

ids = torch.tensor(tok.encode(val_text))
T = 128                                   # 基盤モデルが一度に見られる長さ
n_win = (len(ids) - 1) // T
x = ids[:n_win * T].view(n_win, T)        # (窓の数, T)
y = ids[1:n_win * T + 1].view(n_win, T)   # 1 つずらした正解
total = 0.0
with torch.no_grad():
    for i in range(0, n_win, 32):
        logits, _ = model(x[i:i + 32])    # (B, T, V)
        total += F.cross_entropy(logits.flatten(0, 1), y[i:i + 32].flatten(),
                                 reduction="sum").item()
n_tok = n_win * T
n_char = len(tok.decode(y.flatten().tolist()))
print(f"予測したトークン {n_tok:,} 個 = {n_char:,} 字"
      f"（1 トークンあたり {n_char / n_tok:.2f} 字）")
print(f"1 トークンあたりの損失 {total / n_tok:.3f}"
      f"  パープレキシティ {math.exp(total / n_tok):.1f}")
print(f"1 文字あたりの損失     {total / n_char:.3f}"
      f"  パープレキシティ {math.exp(total / n_char):.1f}")
