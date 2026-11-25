"""第 7 章（図の追加）: 語彙を大きくしたときの「代償」を測るスイープ。

本書のトークナイザ（data/bpe4096.json）のマージ規則を、先頭から
k 個だけ使う（語彙 256 + k）。語彙を 256〜4,088 の 9 通りに変え、
(1) 語彙の中身の内訳（バイト・断片・1 字・2 字・3 字以上）
(2) 小説の訓練用の部分を切ったとき、1 種類のトークンが何回現れるか
(3) 基盤モデルと同じ形（d=192、4 層）で、トークンごとの数
    （埋め込み。出力の重みと共有）が全パラメータに占める割合
を数える。変えるのは語彙の大きさだけで、文章とモデルの形は固定する。
乱数は使わない（モデルは形を作るだけで、学習しない）。
    cd code && python ch07/sweep_vocab_cost.py > ch07/out/sweep_vocab_cost.txt
出力: ch07/out/sweep_vocab_cost.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.model import GPT, GPTConfig
from llmbook.tokenizer import BPETokenizer, _merge_pair_np

OUT = Path(__file__).parent / "out"
LEVELS = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4088]
D = 192  # 基盤モデルの埋め込み次元 d

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
train_text, _ = train_val_split(load_corpus())
ids = np.frombuffer(train_text.encode("utf-8"),
                    dtype=np.uint8).astype(np.int64)


def kind(i: int) -> str:
    """トークン i の種類。バイト 1 個、断片、文字数。"""
    if i < 256:
        return "byte"
    try:
        n = len(tok.vocab[i].decode("utf-8"))
    except UnicodeDecodeError:
        return "frag"  # 文字として読めないバイトの並び
    return {1: "c1", 2: "c2"}.get(n, "c3")  # 3 字以上はまとめる


def n_params(vocab: int) -> int:
    """基盤モデルと同じ形で、語彙だけ変えたモデルのパラメータ数。"""
    cfg = GPTConfig(vocab_size=vocab, block_size=128, n_layer=4,
                    n_head=4, n_embd=D)
    return sum(p.numel() for p in GPT(cfg).parameters())  # 共有は 1 回


rules = list(tok.merges.items())
rows = []
done = 0  # ここまでに当てはめた規則の数
for V in LEVELS:
    for (a, b), new_id in rules[done:V - 256]:  # 規則を学習した順に足す
        ids = _merge_pair_np(ids, a, b, new_id)
    done = V - 256
    freq = np.bincount(ids, minlength=V)  # 各トークンの出現回数
    seen = freq[freq > 0]  # 一度でも現れるトークン
    kinds = [kind(i) for i in range(V)]
    cnt = {k: kinds.count(k) for k in ["byte", "frag", "c1", "c2", "c3"]}
    emb, total = V * D, n_params(V)
    rows.append([V, len(ids), len(seen), int(np.median(seen)),
                 float((seen < 100).mean()), cnt["byte"], cnt["frag"],
                 cnt["c1"], cnt["c2"], cnt["c3"], emb, total, emb / total])
    print(f"語彙 {V:4d}: 列 {len(ids):>9,} 個、現れる種類 {len(seen):4d}、"
          f"回数の中央値 {int(np.median(seen)):>6,}、"
          f"100 回未満 {(seen < 100).mean():5.1%}、"
          f"埋め込み {emb:>7,} / 全体 {total:>9,}（{emb / total:.1%}）")

header = ["vocab", "n_tokens", "n_seen", "median", "frac_lt100", "byte",
          "frag", "c1", "c2", "c3", "emb", "total", "emb_share"]
with open(OUT / "sweep_vocab_cost.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in rows:
        w.writerow(r[:4] + [f"{r[4]:.4f}"] + r[5:12] + [f"{r[12]:.4f}"])
print("内訳（バイト / 断片 / 1 字 / 2 字 / 3 字以上）")
for r in rows:
    print(f"語彙 {r[0]:4d}: " + " / ".join(str(x) for x in r[5:10]))
