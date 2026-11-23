# 本書のコード

『ゼロから作る大規模言語モデル』のサンプルコードです。GPU のない PC で動きます。

くわしい手順（環境の作り方、成功の印、うまくいかないときの対処）は、本書の付録 B にあります。要点だけ書くと、次のとおりです。

```bash
python -m venv .venv && source .venv/bin/activate   # 仮想環境（Windows は .venv\Scripts\activate）
pip install torch numpy                              # ほぼすべてのコードは、この二つだけで動く
cd code                                              # すべてここから実行する
export PYTHONPATH=$PWD                               # llmbook を見つけられるようにする
python scripts/prepare_data.py                # コーパス取得 → BPE 学習 → data/*.bin   （3〜5 分）
python scripts/pretrain_base.py               # 基盤モデルの事前学習 → checkpoints/base.pt（30〜40 分）
python scripts/sft_chat.py                    # 計算アシスタントへの SFT → checkpoints/sft.pt（約 2 分）
python ch00/hello_llm.py                      # 動作確認
```

`data/` と `checkpoints/` の中身（コーパス、トークナイザ、学習済みのモデル）は、このリポジトリに入っていません。上の 3 本の共有スクリプトで、自分の PC で作ります。
各章の `chNN/out/` には、本文に載せた実行結果が入っています。スクリプトを動かすと上書きされます。

## ディレクトリ

| 場所 | 内容 |
|---|---|
| `llmbook/` | 複数の章で共有するライブラリ |
| `scripts/` | 共有するデータとモデルを作るスクリプト |
| `data/` | `soseki.txt`（コーパス）, `bpe4096.json`（トークナイザ）, `train.bin`, `val.bin`（トークン列, uint16） |
| `checkpoints/` | `base.pt`, `base_log.csv`, `sft.pt`、各章の `chNN_*.pt` |
| `chNN/` | 第 NN 章のスクリプト。実行結果は `chNN/out/` |

## 共有ライブラリ `llmbook` の API

掲載用のスニペット印（`# [snippet: 名前]`）が付いています。本文では
`\snippet{キャプション}{code:label}{llmbook/model.py}{attention}` のように取り込めます。
印の一覧: `grep -n "snippet:" llmbook/*.py`

### `llmbook.corpus`
- `DATA_DIR`: `code/data` の `Path`。チェックポイントは `DATA_DIR.parent / "checkpoints"`。
- `load_corpus() -> str`: 漱石 14 作品を整形して連結した文字列（2,407,914 字、異なり 4,078 字）。作品の間は改行で区切られる。
- `train_val_split(text, val_ratio=0.05) -> (train_text, val_text)`: 末尾 5% を検証用にする。
- `WORKS`, `download_work`, `clean_aozora`, `build_corpus`: 青空文庫からの取得と整形（ルビ・注記・ヘッダ・フッタの除去）。

### `llmbook.tokenizer`
- `SPECIAL_TOKENS`: `<|endoftext|>`(4088), `<|user|>`(4089), `<|assistant|>`(4090), `<|end|>`(4091), `<|tool|>`(4092), `<|result|>`(4093), `<|think|>`(4094), `<|pad|>`(4095)。
- `CharTokenizer(text)`: `.encode(str) -> list[int]`, `.decode(ids) -> str`, `.vocab_size`。
- `count_pairs(ids)`, `merge_pair(ids, pair, new_id)`: BPE の素朴版の部品。
- `BPETokenizer`: `.train(text, vocab_size, special_tokens=SPECIAL_TOKENS, verbose=False)`, `.encode(str) -> list[int]`（特殊トークンは 1 トークンになる）,
  `.decode(ids) -> str`, `.token_str(i)`, `.save(path)`, `BPETokenizer.load(path)`, `.vocab_size`, `.merges`（`{(a, b): new_id}`）, `.vocab`（`{id: bytes}`）, `.special_tokens`（`{名前: id}`）。
- 学習済み: `BPETokenizer.load(DATA_DIR / "bpe4096.json")`。語彙 4096 = 256 バイト + 3832 マージ + 特殊 8。
  漱石コーパスで 1 トークンあたり約 1.68 字。計算ドリルの数字は 1 桁 1 トークン（数字どうしのマージは学習されていない。
  `+`, `-`, `=`, 改行も 1 バイト 1 トークン）。

### `llmbook.model`
- `GPTConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0)`（既定値。基盤モデルは `n_embd=192`）
- `GPT(config)`: `.forward(idx, targets=None) -> (logits, loss)`。`idx` は `(B, T)` の long。`logits` は `(B, T, V)`。
  `targets` の `-100` は損失から除外。`.generate(idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_token=None)`
  （`temperature=0.0` で貪欲）。`.num_params()`。部品: `.tok_emb`, `.pos_emb`, `.blocks[i].attn` / `.mlp` / `.ln1` / `.ln2`, `.ln_f`, `.lm_head`（`tok_emb` と重み共有）。
- `CausalSelfAttention`: `.qkv`（`Linear(d, 3d)`）, `.proj`（`Linear(d, d)`）。`MLP`: `.fc`（`Linear(d, 4d)`）, `.proj`（`Linear(4d, d)`）。全 `Linear` はバイアスなし。
- 構造は GPT-2 と同じ（Pre-LN、学習型の位置埋め込み、GELU、重み共有）。

### `llmbook.train`
- `get_batch(data, batch_size, block_size, device="cpu") -> (x, y)`: 1 次元トークン列からランダムな窓を切り出す。`y` は `x` を 1 つずらしたもの。
- `estimate_loss(model, data, batch_size, n_batches=20, device="cpu") -> float`
- `lr_at(step, max_lr, warmup_steps, max_steps, min_lr_ratio=0.1) -> float`: 線形ウォームアップ + コサイン減衰。
- `make_optimizer(model, lr, weight_decay=0.1)`: AdamW（β=(0.9, 0.95)、重み減衰は 2 次元以上のパラメータのみ、`requires_grad=True` のものだけ）。
- `train_lm(model, train_data, val_data, max_steps, batch_size=32, max_lr=1e-3, warmup_steps=100, weight_decay=0.1, grad_clip=1.0, eval_interval=250, device="cpu", log_path=None) -> list[dict]`
- `save_checkpoint(model, path, **extra)`, `load_checkpoint(path, device="cpu") -> GPT`

データの読み方:
```python
import numpy as np, torch
from llmbook.corpus import DATA_DIR
train_data = torch.from_numpy(np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16).astype(np.int64))
```

### `llmbook.arith`
- `make_problem(rng, max_value=999, ops="+-") -> ("12+35=", "47")`: `rng` は `random.Random`。桁数（1〜3 桁）を先に選ぶ。引き算の答えは非負。`ops` に `"*"` も指定可。
- `drill_text(n_lines, seed=0, ...) -> str`: `12+35=47\n` 形式の行。
- `complete(model, tok, prompt, max_new_tokens=8, stop="\n", device="cpu") -> str`: 貪欲生成して `stop` の手前まで返す。
- `drill_accuracy(model, tok, n_problems=200, seed=12345, max_value=999, ops="+-", device="cpu") -> float`: `"\n12+35="` を与えて正答率を測る。

### `llmbook.chat`
- `render_chat(messages, add_generation_prompt=False) -> str`: `[{"role": "user", "content": "12+35は？"}, ...]` → `<|user|>12+35は？<|end|>...`
- `QUESTION_TEMPLATES`: `+`, `-`, `*` それぞれ 3 通りの質問文。
- `make_chat_example(rng, max_value=999, ops="+-") -> (messages, answer)`: 模範応答は `f"{answer}です。"`。
- `encode_chat_example(tok, messages, learn_role=False) -> (ids, labels)`: `labels` は次トークン。モデルが書くべき発話（user と result 以外）の本文と `<|end|>` だけが正解で、ほかは `IGNORE`(-100)。
- `collate(examples, pad_id) -> (x, y)`: 右詰めパディングしてバッチ化。
- `chat_reply(model, tok, question, max_new_tokens=24, temperature=0.0, device="cpu") -> str`
- `chat_accuracy(model, tok, n_problems=200, seed=12345, max_value=999, ops="+-", device="cpu") -> float`: 応答が `"{answer}です。"` と完全一致する割合。

## 共有資産の測定値

測定は執筆環境（arm64 の CPU 4 コア、GPU なし、他の作業と共有）でのものです。本文の数値は、この値を使っています。
手元で動かすと、PC や PyTorch の版の違いで、少し違う値になることがあります（付録 B の「数値が本文と違うとき」）。

### 基盤モデル `checkpoints/base.pt`（`scripts/pretrain_base.py`）
- 構成: `GPTConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4, n_embd=192)`、パラメータ数 **2,583,936**（約 260 万）。
- データ: `train.bin` 2,479,113 トークン（漱石 + 計算ドリル 12 万行を混ぜたもの）、`val.bin` 115,204 トークン。
- 学習: 4,000 ステップ、バッチ 32、系列長 128（1 ステップ 4,096 トークン、合計 約 1,640 万トークン ≒ 6.6 エポック）、
  AdamW、最大学習率 1e-3、ウォームアップ 200 ステップ + コサイン減衰、勾配クリップ 1.0。
- 所要時間: **約 36 分**（2,163 秒。混雑時の値。空いていれば 20 分前後）。本文では「30〜40 分」と書いています。
- 損失の推移（`checkpoints/base_log.csv`。列は `step,train_loss,val_loss,time`。500 ステップごと、各 20 バッチの平均）:

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| train | 8.342 | 4.412 | 3.761 | 3.620 | 3.553 | 3.429 | 3.184 | 3.117 | 3.108 |
| val | 8.343 | 4.348 | 4.120 | 3.888 | 3.714 | 3.599 | 3.429 | 3.392 | 3.478 |

  初期値 8.34 ≒ ln 4096 = 8.318（一様分布の交差エントロピー）。最後は train 3.11 / val 3.4〜3.5 で、軽い過学習が始まっている。
- 生成例（`"\n　吾輩は"` から、温度 0.8、top-k 50）:
  `吾輩はまた席を立てて「どうだい」と答えた。すると突然怒鳴るい事をしなかったと云う声がした。「御母さん」と御馳走したのは君も…`
  日本語らしいが意味は怪しい、という水準。**有名な書き出しを暗記してはいない**（「吾輩は猫」の次に「である」は上位に来ない）。
- 次トークン予測の例: `"どうも仕方が"` → `ない。` 0.559, `ないから` 0.121, `ないと` 0.064, `ないの` 0.045, `ない` 0.016（第 0 章の図）。
  注意: 「である」「ない。」のように、よく出る並びは 1 トークンにまとまっている。そのため `"吾輩は猫で"` の次に `"ある"` は来ない
  （「で」単独のトークンの後に「ある」が続く並びは、学習データにほぼ存在しない）。
- **計算ドリル正答率（貪欲、1〜3 桁の加減算 200 問、`drill_accuracy` の既定値）: 86.5%**。
  例: `"\n12+35="` の次トークンは `5` 0.559, `4` 0.428 で、**貪欲だと `57` と誤答する**（温度 0.8 のサンプリングでは `47` も出る）。
  `"\n3+4="` → `7` 0.898。`"\n25+48="` → `7` 0.682（正解の十の位）。繰り上がり・桁数の多い問題で誤りが多い。
  掛け算（`*`）は学習データに無いので解けない。
- 第 0 章の実行例（`ch00/hello_llm.py`, seed 0）では `12+35=47` に続けて `2+91=93`, `7-1=6`, `885-4=881`, `83+7=90` と自分で問題を作って解く。

### 計算アシスタント `checkpoints/sft.pt`（`scripts/sft_chat.py`）
- `base.pt` から、対話形式の 3,000 例（1〜3 桁の加減算、質問文 3 通りずつ）で 600 ステップ、バッチ 32、学習率 3e-4。所要 **約 1 分 40 秒**。
- 損失: step 0 で 8.97 → 100 で 1.05 → 300 で 0.16 → 599 で 0.07。
- **対話正答率（`chat_accuracy` の既定値、200 問）: SFT 前 0.0% → SFT 後 87.0%**。
- SFT 前の基盤モデルに `<|user|>12+35は？<|end|><|assistant|>` を与えると、`262=540\n9+3=12\n9+8=17\n9+` のようにドリルの続きを書き出す
  （質問に答えるという振る舞いを知らない）。
- 注意: 第 16 章は同じ手順を `code/ch16/` で作り直しており、そちらのモデル `checkpoints/ch16_sft.pt` は 89.0% になる
  （データ・乱数の種・ハイパーパラメータは同じ。損失は 100 ステップ目まで一致し、その後ずれる。CPU のスレッド数の違いによる丸め誤差が増幅されたと考えられる。第 16 章のコラム参照。200 問の評価では ±2〜3 ポイントは誤差の範囲）。**第 17 章以降が出発点・教師として使うのは `sft.pt`（87.0%）**。
- SFT 後: `12+35は？` → `47です。`、`300から123を引くといくつですか？` → `177です。`、`7足す8は？` → `15です。`

### 物語の流れ（章をまたいで使う数値）
| 段階 | 章 | モデル | 正答率 |
|---|---|---|---|
| 事前学習 | 13 | `base.pt` | ドリル形式 86.5%（200 問）／85.1%（1,000 問）、対話形式 0% |
| SFT | 16 | `sft.pt` | 対話形式 87.0%（200 問）／**82.6%（1,000 問、seed 777）** |
| 強化学習（電卓なし） | 22 | `ch22_rl.pt` | 同じ 1,000 問で **84.7%**（正解つきの SFT を続けても 88.6% 止まり） |
| ツール SFT | 23 | `ch23_tool_sft.pt` | 「a+bは？」型だけの 100 問で加減算 81%（電卓 0%）、掛け算 88%、掛けて足す 99%。第 16 章の 3 通りの言い回しの 1,000 問では加減算 **31.6%**（電卓 42.8%。電卓を使うと式を `*` で書き誤る） |
| Agentic RL | 24 | `ch24_agentic_rl.pt` | コールドスタート SFT（63.9%）→ GRPO で、同じ 1,000 問の加減算 **92.8%**（電卓 100%）、掛け算 89.0%、掛けて足す 98.0%（各 300 問） |

物語の筋: 能力そのものを伸ばす強化学習（第 22 章）は、260 万パラメータのモデルでは伸びが小さい（82.6% → 84.7%）。
一方、「いつ電卓を使うか」という振る舞いを変える強化学習（第 24 章）は大きく効く（→ 92.8%）。
200 問の評価は ±2〜3 ポイントぶれるので、比較には 1,000 問の値を使う。掛け算は電卓ツール（第 23 章）で初めて解けるようになる。
