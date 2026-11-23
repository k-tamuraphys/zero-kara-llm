"""評価済みの軌跡（out/eval_*.csv）を、質問の言い回しと
「電卓を使ったか」で集計する。モデルは動かさない（数秒）。
  python phrasing.py  → out/phrasing.txt
"""
import csv, re
from pathlib import Path

OUT_DIR = Path(__file__).parent / "out"
RUNS = [("before", "学習前"), ("c0", "RLのみc=0"), ("c02", "RLのみc=0.2"),
        ("cold", "SFT後"), ("cold_c0", "SFT+RLc=0"),
        ("cold_c02", "SFT+RLc=0.2"), ("cold_c035", "SFT+RLc=0.35"),
        ("cold_c05", "SFT+RLc=0.5")]
TYPES = ["a+bは？", "a足すbは？", "aとbを足すと…"]


def phrasing(q):
    """質問文の言い回し（第 16 章の 3 通り）。"""
    if re.search(r"\d[+\-]\d", q):
        return TYPES[0]
    return TYPES[1] if q.endswith("は？") else TYPES[2]


def bucket(q):
    """analyze.py と同じ分け方（大きいほうの桁数、繰り上がりの有無）。"""
    a, b = map(int, re.findall(r"\d+", q)[:2])
    sub = "-" in q or "引" in q
    carry, x, y, borrow = False, a, b, 0
    while x or y:
        dx, dy = x % 10, y % 10
        over = (dx - dy - borrow < 0) if sub else (dx + dy + borrow >= 10)
        carry, borrow = carry or over, int(over)
        x, y = x // 10, y // 10
    return f"{len(str(max(a, b)))}桁・{'あり' if carry else 'なし'}"


def rate(rs):
    """(問題数, 正答率)。"""
    return len(rs), sum(int(r["correct"]) for r in rs) / max(1, len(rs))


lines = []
p = lines.append
rows = {}
for name, _ in RUNS:
    with open(OUT_DIR / f"eval_{name}.csv") as f:
        rows[name] = [r for r in csv.DictReader(f) if r["kind"] == "add"]

p("加減算 1,000 問（貪欲）：言い回し別の電卓使用率（正答率）")
for name, label in RUNS:
    cells = []
    for t in TYPES:
        rs = [r for r in rows[name] if phrasing(r["question"]) == t]
        tool = sum(int(r["calls"]) > 0 for r in rs) / len(rs)
        cells.append(f"{t} {tool:.2f}（{rate(rs)[1]:.2f}）")
    p(f"  {label:<13}" + "  ".join(cells))

p("電卓を使った問題／暗算した問題の、問題数と正答率")
for name, label in RUNS:
    used = [r for r in rows[name] if int(r["calls"]) > 0]
    none = [r for r in rows[name] if int(r["calls"]) == 0]
    (nu, au), (nn, an) = rate(used), rate(none)
    p(f"  {label:<13}電卓 {nu:>4} 問 {au:.3f}   暗算 {nn:>4} 問 {an:.3f}")

p("「a+bは？」型以外の質問で、暗算した問題の正答率")
for name, label in RUNS[:3]:
    rs = [r for r in rows[name] if int(r["calls"]) == 0
          and phrasing(r["question"]) != TYPES[0]]
    p(f"  {label:<13}{rate(rs)[0]:>4} 問 {rate(rs)[1]:.3f}")

p("SFT+RLc=0.35 が暗算した問題（種類別）")
none = [r for r in rows["cold_c035"] if int(r["calls"]) == 0]
for b in sorted({bucket(r["question"]) for r in rows["cold_c035"]}):
    rs = [r for r in none if bucket(r["question"]) == b]
    n, acc = rate(rs)
    p(f"  {b}: {n:>3} 問  正解 {round(n * acc):>2} 問")
n_sym = sum(phrasing(r["question"]) == TYPES[0] for r in none)
p(f"  合計 {len(none)} 問（うち「a+bは？」型 {n_sym} 問）")

text = "\n".join(lines)
print(text)
(OUT_DIR / "phrasing.txt").write_text(text + "\n")
