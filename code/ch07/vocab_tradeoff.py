"""語彙サイズと列の長さの関係を測る。

学習済みトークナイザのマージ規則を 1 個目から順に再生し、そのたびに
「その組が何回あったか」「列が何個になったか」を記録する。
k 個目まで再生した状態は、語彙サイズ 256 + k のトークナイザと同じ。

実行:  python ch07/vocab_tradeoff.py   （code/ ディレクトリで実行。約 1 分）
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ch07.capture import OUT, Capture
from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.tokenizer import BPETokenizer, _merge_pair_np, count_pairs
from llmbook.tokenizer import merge_pair

cap = Capture()
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
train_text, val_text = train_val_split(load_corpus())


def to_array(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


train, val = to_array(train_text), to_array(val_text)

# ---- 素朴版と NumPy 版の、マージ 1 回ぶんの時間（最初の 1 回、3 回測る）----
ids_list = train.tolist()
naive, fast = [], []
for _ in range(3):
    t0 = time.perf_counter()
    counts = count_pairs(ids_list)
    pair = max(counts, key=counts.get)
    merge_pair(ids_list, pair, 256)
    naive.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    size = 4088
    code = train[:-1] * size + train[1:]
    best = int(np.bincount(code, minlength=size * size).argmax())
    _merge_pair_np(train, *divmod(best, size), 256)
    fast.append(time.perf_counter() - t0)
naive_t, fast_t = sorted(naive)[1], sorted(fast)[1]
n_merges = len(tok.merges)
print(f"列の長さ {len(train):,} バイト、最も多い組 {pair}")
print(f"素朴版  : マージ 1 回 {naive_t:.2f} 秒"
      f" -> {n_merges} 回なら約 {naive_t * n_merges / 3600:.1f} 時間")
print(f"NumPy 版: マージ 1 回 {fast_t:.3f} 秒"
      f" -> {n_merges} 回なら約 {fast_t * n_merges / 60:.1f} 分")
cap.save("speed.txt")
del ids_list

# ---- マージを順に再生する ----------------------------------------------
rows = []
n_char, n_char_val = len(train_text), len(val_text)
rows.append((0, 256, 0, len(train), n_char / len(train),
             n_char_val / len(val)))
for k, ((a, b), new_id) in enumerate(tok.merges.items(), start=1):
    count = int(np.count_nonzero((train[:-1] == a) & (train[1:] == b)))
    train = _merge_pair_np(train, a, b, new_id)
    val = _merge_pair_np(val, a, b, new_id)
    rows.append((k, 256 + k, count, len(train), n_char / len(train),
                 n_char_val / len(val)))

header = ["merge", "vocab", "count", "n_tokens", "chars_per_token",
          "chars_per_token_val"]
with open(OUT / "merge_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.4f}", f"{r[5]:.4f}"])

# 図に使う間引いた版（対数軸で等間隔に近い点）
keep = sorted({int(round(x)) for x in np.geomspace(1, n_merges, 60)})
with open(OUT / "tradeoff.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for k in keep:
        r = rows[k]
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.4f}", f"{r[5]:.4f}"])

print("語彙サイズ  トークン数   字/トークン（訓練）（検証）  "
      "最後のマージの回数")
for k in [0, 256, 768, 1792, 3832]:
    r = rows[k]
    print(f"{r[1]:8d} {r[3]:11,d} {r[4]:14.2f} {r[5]:8.2f} {r[2]:14,d}")
cap.save("tradeoff.txt")
