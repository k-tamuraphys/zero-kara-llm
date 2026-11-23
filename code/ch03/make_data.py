"""漱石コーパスから「文字数 → トークン数」のデータを作る。

小説の本文からランダムな長さ（20〜200 字）の断片を切り出し、
その文字数と、トークナイザで数えたトークン数の組を CSV に保存する。
"""

import random
from pathlib import Path

from llmbook.corpus import DATA_DIR, load_corpus
from llmbook.tokenizer import BPETokenizer

OUT = Path(__file__).parent / "out"
text = load_corpus()  # 漱石 14 作品（約 240 万字）
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")  # 第 7 章で作るトークナイザ


# [snippet: make-pairs]
def make_pairs(n: int, seed: int) -> list[tuple[int, int, str]]:
    """（文字数, トークン数, 断片）の組を n 個作る。"""
    rng = random.Random(seed)  # 乱数の種を固定して、毎回同じデータにする
    rows = []
    for _ in range(n):
        n_chars = rng.randint(20, 200)  # 断片の長さを 20〜200 字から選ぶ
        start = rng.randrange(len(text) - n_chars)  # 切り出す位置
        passage = text[start:start + n_chars]
        n_tokens = len(tok.encode(passage))  # トークンに区切って数える
        rows.append((n_chars, n_tokens, passage))
    return rows
# [end: make-pairs]


def save(rows, name: str) -> None:
    with open(OUT / name, "w", encoding="utf-8") as f:
        f.write("chars,tokens\n")
        for n_chars, n_tokens, _ in rows:
            f.write(f"{n_chars},{n_tokens}\n")


if __name__ == "__main__":
    train = make_pairs(100, seed=0)  # 訓練用 100 件
    test = make_pairs(100, seed=1)  # 学習に使わない 100 件
    big = make_pairs(10000, seed=2)  # ミニバッチの実験用 1 万件
    save(train, "char_token_train.csv")
    save(test, "char_token_test.csv")
    save(big, "char_token_big.csv")
    print("文字数 トークン数  断片（先頭 24 字）")
    for n_chars, n_tokens, passage in train[:6]:
        head = passage.replace("\n", " ")[:24]
        print(f"{n_chars:6d} {n_tokens:8d}  {head}…")
