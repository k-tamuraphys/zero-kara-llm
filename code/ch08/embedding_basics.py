"""第 8 章: 埋め込みは「学習される表引き」である。

実行: cd code && python ch08/embedding_basics.py
出力: ch08/out/basics_*.txt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from capture import Capture
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer

torch.manual_seed(0)
cap = Capture()

# [snippet: ids]
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
for word in ["先生", "奥さん", "東京", "7", "8"]:
    print(word, tok.encode(word))
# [end: ids]
cap.save("basics_ids.txt")

# [snippet: onehot]
V, d = 4096, 4                      # 語彙サイズと、ベクトルの次元（小さめ）
E = torch.randn(V, d)               # (V, d) の表。1 行が 1 トークンのベクトル
x = 692                             # 「先生」の番号
onehot = F.one_hot(torch.tensor(x), V).float()   # (V,) 692 番目だけ 1
print("one-hot の長さ:", onehot.shape[0], " 1 の個数:", int(onehot.sum()))
print("one-hot × 表:", onehot @ E)  # (V,) と (V, d) の積 → (d,)
print("表の 692 行目:", E[x])       # 掛け算をせず、行を取り出すだけ
# [end: onehot]
cap.save("basics_onehot.txt")

# [snippet: embedding]
emb = nn.Embedding(V, d)            # 中身は (V, d) の表（最初は乱数）
ids = torch.tensor([[692, 1617, 1306]])          # (B, n) = (1, 3)
out = emb(ids)                      # 番号の位置に、その行が入る
print("入力の形:", tuple(ids.shape), "→ 出力の形:", tuple(out.shape))
print("表の形:", tuple(emb.weight.shape))
print("emb.weight[692] と同じか:", torch.equal(out[0, 0], emb.weight[692]))
# [end: embedding]
cap.save("basics_embedding.txt")

# [snippet: grad]
loss = out.sum()                    # 仮の損失
loss.backward()
touched = (emb.weight.grad.abs().sum(dim=1) > 0).nonzero().flatten()
print("勾配が 0 でない行:", touched.tolist())
# [end: grad]
cap.save("basics_grad.txt")
