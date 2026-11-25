"""第 7 章（図の追加）: トークナイザの学習に使う文章の量を変えると、
1 トークンあたりの文字数（圧縮の度合い）がどう変わるかを測るスイープ。

小説の訓練用の部分（約 229 万字）の先頭から、1/16, 1/8, 1/4, 1/2, 全部の
5 通りの量を取り、どれも語彙 4,096（マージ 3,832 回）の BPE を学習する。
変えるのは文章の量だけで、語彙の大きさと学習の手順（BPETokenizer.train）
は固定する。全部の量は data/bpe4096.json と同じ手順なので、それを読む。
乱数は使わない（BPE の学習は、同じ文章なら必ず同じ規則になる）。
学習したトークナイザは out/sweep_tok/ に保存し、次からは読むだけにする
（途中で止まっても、続きから走らせ直せる）。
    cd code && python ch07/sweep_tok_data.py > ch07/out/sweep_tok_data.txt
出力: ch07/out/sweep_tok_data.csv（量ごとの訓練側・検証側の文字数）
      ch07/out/sweep_tok_data_vocab.csv（検証用の文字数を、語彙の途中で）
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.tokenizer import BPETokenizer, _merge_pair_np

OUT = Path(__file__).parent / "out"
SAVE = OUT / "sweep_tok"
SAVE.mkdir(parents=True, exist_ok=True)
FRACS = [16, 8, 4, 2, 1]  # 先頭の 1/16, 1/8, 1/4, 1/2, 全部
LEVELS = [512, 1024, 2048, 4088]  # 途中の語彙（特殊トークンを除く）

train_text, val_text = train_val_split(load_corpus())
# 2 作目『坊っちゃん』の書き出しの位置（先頭の何割が 1 作目か）
start2 = train_text.find("親譲りの無鉄砲で")
print(f"訓練用 {len(train_text):,} 字、検証用 {len(val_text):,} 字")
print(f"1 作目『吾輩は猫である』は先頭の {start2:,} 字"
      f"（訓練用の {start2 / len(train_text):.1%}）")


def to_array(text: str) -> np.ndarray:
    """文字列を UTF-8 のバイトの配列にする。"""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def get_tokenizer(frac: int, part: str) -> BPETokenizer:
    """先頭の 1/frac で学習したトークナイザ（保存してあれば読む）。"""
    if frac == 1:
        return BPETokenizer.load(DATA_DIR / "bpe4096.json")
    path = SAVE / f"bpe4096_1of{frac}.json"
    if path.exists():
        return BPETokenizer.load(path)
    t0 = time.perf_counter()
    tok = BPETokenizer()
    tok.train(part, 4096)  # 語彙の大きさは全部の量と同じ
    tok.save(path)
    print(f"  1/{frac} を学習: {time.perf_counter() - t0:.0f} 秒")
    return tok


rows, vrows = [], []
for frac in FRACS:
    part = train_text[:len(train_text) // frac]
    tok = get_tokenizer(frac, part)
    own, val = to_array(part), to_array(val_text)
    cpt_val = {}
    for k, ((a, b), new_id) in enumerate(tok.merges.items(), start=1):
        own = _merge_pair_np(own, a, b, new_id)  # 学習に使った文章
        val = _merge_pair_np(val, a, b, new_id)  # 検証用（見ていない）
        if 256 + k in LEVELS:
            cpt_val[256 + k] = len(val_text) / len(val)
    cpt_own = len(part) / len(own)
    rows.append((frac, len(part), cpt_own, cpt_val[4088]))
    vrows.append([frac] + [cpt_val[v] for v in LEVELS])
    print(f"1/{frac:<2d}（{len(part):>9,} 字）: 字/トークン "
          f"学習に使った文章 {cpt_own:.2f}、検証用 {cpt_val[4088]:.2f}、"
          f"差 {cpt_own - cpt_val[4088]:+.2f}")

with open(OUT / "sweep_tok_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frac", "chars", "man", "cpt_own", "cpt_val"])
    for frac, n, own, val in rows:  # man は万字の単位
        w.writerow([frac, n, f"{n / 1e4:.1f}", f"{own:.4f}", f"{val:.4f}"])
with open(OUT / "sweep_tok_data_vocab.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["vocab"] + [f"val_1of{fr}" for fr in FRACS])
    for j, v in enumerate(LEVELS):  # 行 = 語彙、列 = 量
        w.writerow([v] + [f"{r[1 + j]:.4f}" for r in vrows])
print("検証用の字/トークン（行 = 語彙、列 = 1/16, 1/8, 1/4, 1/2, 全部）")
for j, v in enumerate(LEVELS):
    print(f"{v:5d}: " + "  ".join(f"{r[1 + j]:.2f}" for r in vrows))
