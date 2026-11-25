"""第 6 章（図の追加）: 本章に出てくるモデルの検証損失を、実行結果の
ファイルから集めて 1 つの csv にする（図「本章のモデルの成績」用）。

数値は書き写さず、ch06/out/ の実行結果から読み取る。損失はどれも
1 文字あたり（基盤モデルは 1 トークン = 平均 1.75 文字なので、
preview_gpt.py が 1 文字あたりに直した値を使う）。
    cd code && python ch06/fig_models.py
出力: ch06/out/fig_models.csv（列 k, name, loss, ppl）
"""
import math
import re
from pathlib import Path

OUT = Path(__file__).parent / "out"


def grab(name, pattern):
    """実行結果のファイル name から、pattern の最初の数を取り出す。"""
    text = (OUT / name).read_text(encoding="utf-8")
    return float(re.search(pattern, text).group(1))


ngram = {int(m.group(1)): float(m.group(2)) for m in re.finditer(
    r"^\s*(\d)\s.*?\s([\d.]+)（", (OUT / "ngram_1.txt").read_text(
        encoding="utf-8"), re.M)}             # n ごとの検証損失
B5 = "bigram_count_5.txt"
models = [  # (名前, 損失, パープレキシティ。ファイルにあればそれを使う)
    ("何も知らない", grab(B5, r"一様分布\s+損失 ([\d.]+)"),
     grab(B5, r"一様分布.*パープレキシティ\s+([\d.]+)")),
    ("ユニグラム", grab(B5, r"ユニグラム\s+損失 ([\d.]+)"),
     grab(B5, r"ユニグラム.*パープレキシティ\s+([\d.]+)")),
    ("バイグラム（数える）", grab(B5, r"バイグラム\s+損失 ([\d.]+)"),
     grab(B5, r"バイグラム.*パープレキシティ\s+([\d.]+)")),
    ("バイグラム（勾配降下法）",
     grab("bigram_nn_3.txt", r"step 100 .*検証 ([\d.]+)"), None),
    ("トライグラム（$n=3$）", ngram[3], None),
    ("$n=4$", ngram[4], None),
    ("$n=5$", ngram[5], None),
    ("基盤モデル（Transformer）",
     grab("preview_gpt.txt", r"1 文字あたりの損失\s+([\d.]+)"),
     grab("preview_gpt.txt", r"1 文字あたり.*パープレキシティ\s+([\d.]+)")),
]
rows = ["k,name,loss,ppl"]
for k, (name, loss, ppl) in enumerate(models):
    ppl = math.exp(loss) if ppl is None else ppl
    shown = f"{ppl:.0f}" if ppl >= 1000 else f"{ppl:.1f}"  # 4078 は整数で
    rows.append(f"{k},{{{name}}},{loss:.3f},{shown}")
    print(f"{name:20s} {loss:.3f} {ppl:8.1f}")
(OUT / "fig_models.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
