"""第 0 章の図に書き込む数値を、基盤モデル base.pt から計算する。

出力（chNN/out/fig_data.txt）を、次の図が使う。
  図「コード 0.1 の中を流れる数」: 形と、上位 3 個のスコア・確率
  図「12+35= の分かれ道」: 1 文字目と 2 文字目の確率、選び方ごとの答え
実行: cd code && python ch00/fig_data.py > ch00/out/fig_data.txt
"""

from collections import Counter

import torch

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.manual_seed(0)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()


def show(i: int) -> str:
    """トークン番号を、改行が見える形の文字列にする。"""
    return repr(tok.decode([i]))


@torch.no_grad()
def next_probs(text: str):
    """text の次のトークンについて、スコアと確率（どちらも (V,)）を返す。"""
    ids = torch.tensor([tok.encode(text)])  # (1, T)
    logits, _ = model(ids)  # (1, T, V)
    z = logits[0, -1]  # 最後の位置だけ取り出す (V,)
    return z, torch.softmax(z, dim=-1)


# ---- 1. コード 0.1 の中を流れる数の形
prompt = "どうも仕方が"
ids = torch.tensor([tok.encode(prompt)])
with torch.no_grad():
    logits, _ = model(ids)
z, p = next_probs(prompt)
print("== flow:", repr(prompt))
print("ids", ids.tolist(), "shape", tuple(ids.shape))
print("logits shape", tuple(logits.shape))
print("last-position scores shape", tuple(z.shape))
print(f"probs shape {tuple(p.shape)}  sum {p.sum().item():.3f}")
for i in torch.topk(p, 3).indices.tolist():
    print(f"  {show(i):10} score {z[i].item():6.2f}  prob {p[i].item():.3f}")

# ---- 2. 「12+35=」の 1 文字目と 2 文字目
q = "\n12+35="
print("== branch:", repr(q), "tokens", len(tok.encode(q)))
z, p = next_probs(q)
for i in torch.topk(p, 3).indices.tolist():
    d = tok.decode([i])
    _, p2 = next_probs(q + d)  # その数字を選んだあとの、次の確率
    j = int(p2.argmax())
    print(f"  1st {d!r} {p[i].item():.3f} -> 2nd {show(j)} "
          f"{p2[j].item():.3f}  (path {p[i].item() * p2[j].item():.3f})")

# ---- 3. 選び方ごとの答え（改行の手前までを答えとみなす）
x = torch.tensor([tok.encode(q)])


def answer(**kw) -> str:
    out = model.generate(x, max_new_tokens=4, **kw)
    return tok.decode(out[0, x.shape[1]:].tolist()).split("\n")[0]


print("greedy (temperature=0):", answer(temperature=0.0))
for kw in [dict(temperature=1.0), dict(temperature=0.8, top_k=50)]:
    torch.manual_seed(0)  # 設定ごとに同じくじの出方から始める
    c = Counter(answer(**kw) for _ in range(1000))
    print("sampling", kw, "1000 draws:", c.most_common(4))
