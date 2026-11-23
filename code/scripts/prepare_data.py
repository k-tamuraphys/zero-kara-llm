"""事前学習用データの準備: コーパス取得 → トークナイザ学習 → トークン化。

実行:  python scripts/prepare_data.py   （code/ ディレクトリで実行する）
出力:  data/soseki.txt, data/bpe4096.json, data/train.bin, data/val.bin
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.arith import drill_text
from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.tokenizer import BPETokenizer

VOCAB_SIZE = 4096

# 1. 小説コーパスを取得し、訓練用と検証用に分ける
novel_train, novel_val = train_val_split(load_corpus())

# 2. 小説の訓練部分で BPE トークナイザを学習する（学習済みなら再利用）
tok_path = DATA_DIR / f"bpe{VOCAB_SIZE}.json"
if tok_path.exists():
    tok = BPETokenizer.load(tok_path)
else:
    tok = BPETokenizer()
    tok.train(novel_train, VOCAB_SIZE, verbose=True)
    tok.save(tok_path)

# 3. 計算ドリルを作る。訓練用と検証用で乱数の種を変える
drill_train = drill_text(120_000, seed=0)
drill_val = drill_text(5_000, seed=1)

# 4. トークン化し、16 ビット整数の配列としてファイルに書き出す
eot = "<|endoftext|>"
for name, docs in [("train", [novel_train, drill_train]),
                   ("val", [novel_val, drill_val])]:
    ids = []
    for doc in docs:
        ids.extend(tok.encode(doc))
        ids.append(tok.special_tokens[eot])  # 文書の区切り
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(DATA_DIR / f"{name}.bin")
    print(f"{name}: {len(arr):,} トークン")
