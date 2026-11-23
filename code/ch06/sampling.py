"""第 6 章: 次の文字の選び方（貪欲、温度、top-k、top-p）を比べる。

題材は、直前の 2 文字から次の文字を予測するトライグラムモデル（数えて作る）。
実行: cd code && python ch06/sampling.py
出力: 標準出力（"---" で区切る）, ch06/out/temperature.csv
"""
from collections import Counter, defaultdict
from pathlib import Path

import torch

from llmbook.corpus import load_corpus, train_val_split

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)

text = load_corpus()
train_text, val_text = train_val_split(text)
chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

table = defaultdict(Counter)  # table[直前の 2 文字][次の文字] = 回数
for i in range(2, len(train_text)):
    table[train_text[i - 2:i]][train_text[i]] += 1


def next_probs(context):
    """直前の 2 文字 context に続く文字の確率分布 (V,)。"""
    p = torch.zeros(V)
    for ch, c in table[context].items():
        p[stoi[ch]] = c
    return p / p.sum()


# [snippet: sample-next]
def sample_next(p, g, temperature=1.0, top_k=None, top_p=None):
    """確率分布 p (V,) を加工してから、次のトークンの番号を 1 個選ぶ。"""
    if temperature == 0:                      # 温度 0 は貪欲: 最大のものを選ぶ
        return int(torch.argmax(p))
    # 温度: ロジット（ここでは log p）を温度で割ってから softmax をかけ直す
    p = torch.softmax(torch.log(p) / temperature, dim=-1)
    if top_k is not None:                     # top-k: 上位 k 個以外を 0 にする
        kth = torch.topk(p, top_k).values[-1]           # k 番目の確率
        p = torch.where(p >= kth, p, torch.zeros_like(p))
    if top_p is not None:                     # top-p: 上から足して p に届く分
        sorted_p, order = torch.sort(p, descending=True)
        before = torch.cumsum(sorted_p, dim=0) - sorted_p  # 自分より上の合計
        keep = order[before < top_p]
        mask = torch.zeros_like(p)
        mask[keep] = 1.0
        p = p * mask
    p = p / p.sum()                           # 合計が 1 になるよう割り直す
    return int(torch.multinomial(p, num_samples=1, generator=g))
# [end: sample-next]


# [snippet: generate]
def generate(start, length, seed=0, **how):
    """start に続けて、1 文字ずつ length 文字を書き足す（自己回帰生成）。"""
    g = torch.Generator().manual_seed(seed)
    out = start
    for _ in range(length):
        p = next_probs(out[-2:])              # 末尾の 2 文字から次を予測する
        out += chars[sample_next(p, g, **how)]  # 選んだ文字を末尾に足して次へ
    return out
# [end: generate]


# 文脈「先生」の次の文字の分布と、温度による変化
p = next_probs("先生")
top = torch.topk(p, 8)
print(f"「先生」の次の文字: 候補 {(p > 0).sum().item()} 種類")
print("       " + "      ".join(chars[i] for i in top.indices))
rows = ["k,t05,t10,t20"]
dists = {}
for t in [0.5, 1.0, 2.0]:
    q = torch.softmax(torch.log(p) / t, dim=-1)
    dists[t] = q
    ent = -(q[q > 0] * torch.log(q[q > 0])).sum().item()
    print(f"温度 {t}  " + "  ".join(f"{q[i].item():.3f}" for i in top.indices)
          + f"   エントロピー {ent:.2f}")
for k, i in enumerate(top.indices):
    rows.append(f"{k}," + ",".join(
        f"{dists[t][i].item():.4f}" for t in [0.5, 1.0, 2.0]))
(OUT / "temperature.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
sorted_p = torch.sort(p, descending=True).values
cum = torch.cumsum(sorted_p, dim=0)
print("上から足した合計  " + "  ".join(f"{c:.3f}" for c in cum[:8]))
print(f"top-p = 0.8 で残る候補 {int((cum - sorted_p < 0.8).sum())} 種類")
print("---")

settings = [("貪欲      ", dict(temperature=0)),
            ("温度 0.5  ", dict(temperature=0.5)),
            ("温度 1.0  ", dict(temperature=1.0)),
            ("温度 1.5  ", dict(temperature=1.5)),
            ("top-k 5   ", dict(top_k=5)),
            ("top-p 0.8 ", dict(top_p=0.8))]
for name, how in settings:
    s = generate("吾輩は", 30, **how)
    print(name + s.replace("\n", "／"))
print("---")
# 同じ設定（温度 1.0）でも、乱数の種が違えば別の文章になる
for seed in [1, 2, 3]:
    s = generate("吾輩は", 30, seed=seed)
    print(f"種 {seed}  " + s.replace("\n", "／"))
