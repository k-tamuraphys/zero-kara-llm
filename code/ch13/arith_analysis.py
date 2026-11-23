"""基盤モデル base.pt の計算ドリルの誤りを、桁数と繰り上がりで分けて調べる。

実行: python ch13/arith_analysis.py  （約 1 分）
出力: ch13/out/arith_analysis.txt, ch13/out/arith_by_kind.csv
"""

import random

import torch

from llmbook.arith import complete, make_problem
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()

# [snippet: next-digit]
prompt = torch.tensor([tok.encode("\n12+35=")])
with torch.no_grad():
    logits, _ = model(prompt)
probs = torch.softmax(logits[0, -1], dim=-1)  # 最後の位置の次トークン分布
top = torch.topk(probs, 3)
for p, i in zip(top.values, top.indices):
    print(f"  {tok.token_str(i.item())!r}: {p.item():.3f}")
# [end: next-digit]


# [snippet: classify]
def kind_of(question: str) -> tuple[str, str]:
    """問題を（演算と最大桁数, 繰り上がり・繰り下がりの有無）に分類する。"""
    op = "+" if "+" in question else "-"
    a, b = (int(s) for s in question[:-1].split(op))
    digits = max(len(str(a)), len(str(b)))
    da, db = str(a)[::-1], str(b)[::-1]  # 1 の位から見る
    carry = False
    for k in range(digits):
        x = int(da[k]) if k < len(da) else 0
        y = int(db[k]) if k < len(db) else 0
        carry |= (x + y >= 10) if op == "+" else (x < y)
    return f"{op}{digits}桁", "あり" if carry else "なし"
# [end: classify]


rng = random.Random(2024)
stats = {}
wrong = []
for _ in range(1000):
    question, answer = make_problem(rng)
    output = complete(model, tok, "\n" + question)
    key = kind_of(question)
    n, c = stats.get(key, (0, 0))
    stats[key] = (n + 1, c + (output == answer))
    if output != answer and len(wrong) < 8:
        wrong.append(f"{question}{output}（正解 {answer}）")

total = sum(n for n, _ in stats.values())
correct = sum(c for _, c in stats.values())
print(f"全体: {correct}/{total} = {correct / total:.1%}")
with open("ch13/out/arith_by_kind.csv", "w") as f:
    f.write("kind,carry,n,acc\n")
    for key in sorted(stats):
        n, c = stats[key]
        f.write(f"{key[0]},{key[1]},{n},{c / n:.3f}\n")
        print(f"{key[0]} 繰り上がり{key[1]}: {c:4d}/{n:4d} = {c / n:.1%}")
print("誤答の例:", " / ".join(wrong[:6]))
