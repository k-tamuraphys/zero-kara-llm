# ゼロから作る大規模言語モデル

**機械学習の基礎から自作 LLM・AI エージェントまで**

機械学習をまったく知らない人が、一冊を読み通すだけで、自分の手で大規模言語モデル（LLM）を作り、育て、AI エージェントに仕立てられるようになること。それを目標にした教科書の、本文とサンプルコードです。

- 数式・図・動くコードの三つを対応づけて説明します。式変形には 1 行ずつ理由を添えます。
- コードはすべて、**GPU のない普通の PC** で動きます（PyTorch と NumPy だけ）。
- 夏目漱石の小説と計算ドリルを読ませた、約 260 万パラメータの小さな GPT を一つ育て、その同じモデルを最後まで使い続けます。事前学習、ファインチューニング、LoRA、蒸留、強化学習、エージェントまで、行う工程は実際の LLM 開発と同じです。

## 読む

組版済みの PDF（B5、約 1,040 ページ）は、[Releases](https://github.com/k-tamuraphys/zero-kara-llm/releases) にあります。

| 部 | 章 |
|---|---|
| | 0 はじめに：本書の歩き方 |
| I 準備：数学と道具 | 1 数学の準備 ／ 2 道具の準備（Python・NumPy・PyTorch） |
| II 機械学習とニューラルネットワーク | 3 機械学習の枠組み ／ 4 ニューラルネットワークと誤差逆伝播法 ／ 5 深いネットワークを学習させる技術 |
| III 言語モデルと Transformer | 6 言語モデル ／ 7 トークン化 ／ 8 埋め込みとニューラル言語モデル ／ 9 注意機構 ／ 10 Transformer：ミニGPT を組み立てる ／ 11 現代の LLM アーキテクチャ |
| IV 大規模に学習させる | 12 なぜ GPU が必要か ／ 13 事前学習の実際 ／ 14 分散学習 ／ 15 計算量の見積もりとインフラ選定 |
| V 事後学習 | 16 事後学習の全体像とファインチューニング ／ 17 LoRA ／ 18 蒸留と量子化 ／ 19 評価と推論 |
| VI 強化学習 | 20 強化学習の基礎 ／ 21 方策勾配法 ／ 22 LLM の強化学習（RLHF・DPO・GRPO） |
| VII AI エージェント | 23 AI エージェントを作る ／ 24 Agentic RL ／ 25 総仕上げ |
| 付録 | A 記号一覧 ／ B 環境構築とコードの動かし方 ／ C 用語集 ／ D 演習問題の略解 ／ E ベンチマーク案内 ／ F 技術選定の地図 |

### どこから読むか

前から全部読む必要はありません。本書は 3 周に分けて読むことを勧めています（くわしくは第 0 章）。

| 周 | ねらい | 章 |
|---|---|---|
| 1 周目（背骨） | ミニGPT を自分で組み立てて学習させ、計算アシスタントに育て、電卓を使うエージェントにする | 0, 3, 4, 6, 8, 9, 10, 13, 16, 23 |
| 2 周目（実用にする） | 学習を安定させ、速くし、安く調整し、測り、強化学習で磨く | 5, 7, 11, 12, 17, 19, 20, 21, 22, 25 |
| 3 周目（大きくする・運用する） | 分散学習、費用の見積もり、圧縮、エージェントの強化学習、道具選び | 14, 15, 18, 24, 付録 E・F |

第 1 章（数学）と第 2 章（PyTorch）は、必要な節だけ読んで、あとは辞書として使う章です。
各章の冒頭には「この章の読み方」の枠があり、節ごとに「必読」「流し読み可」「後回し可」を示してあります。

## コードを動かす

Python 3.11 以降があれば動きます。くわしい手順（成功の印、うまくいかないときの対処）は付録 B に、ライブラリの API は [`code/README.md`](code/README.md) にあります。

```bash
git clone https://github.com/k-tamuraphys/zero-kara-llm.git
cd zero-kara-llm
python -m venv .venv && source .venv/bin/activate   # Windows は .venv\Scripts\activate
pip install torch numpy
cd code
export PYTHONPATH=$PWD
python scripts/prepare_data.py     # 文章を取得してトークン化する（3〜5 分）
python scripts/pretrain_base.py    # 基盤モデルを学習させる（30〜40 分）
python scripts/sft_chat.py         # 計算アシスタントにする（約 2 分）
python ch00/hello_llm.py           # 動作を確かめる
```

時間は、4 コアの CPU（arm64）、メモリ 8 GB の環境で測ったものです。

学習に使う文章と、学習済みのモデルは、このリポジトリに入っていません。文章は `prepare_data.py` が[青空文庫](https://www.aozora.gr.jp/)から取得し、モデルは上の手順で自分の PC で作ります。
各章の `code/chNN/out/` には、本文に載せた実行結果が入っています。

## 本を組版する

必要なものは、TeX Live（LuaLaTeX、原ノ味フォント、`bibtex`、`upmendex`）と Python 3 です。

```bash
tools/build.sh                        # 本全体 → build/main.pdf（3 回組むので、10 分以上かかります）
tools/build_chapter.sh ch10_transformer   # 1 章だけ → build/ch10_transformer.pdf（他章への参照は「??」になります）
```

`lualatex` や `python` が PATH にない環境では、`tools/env.local.sh` を作り、そこで PATH を足してください（`tools/env.sh` が読み込みます）。
本文に載せるコードは、`tools/extract_snippets.py` が `code/` のファイルから切り出します。本文とコードが食い違わないようにするためです。

## ディレクトリ

| 場所 | 内容 |
|---|---|
| `chapters/` | 本文（LaTeX）。`chNN_*.tex` が各章、`appX_*.tex` が付録、`ansNN.tex` が演習問題の略解 |
| `main.tex`, `preamble.tex`, `refs.bib` | 本全体の入口、共通の設定とマクロ、参考文献 |
| `code/` | サンプルコード。`llmbook/` が共有ライブラリ、`scripts/` がデータとモデルを作るスクリプト、`chNN/` が各章のスクリプトと実行結果 |
| `tools/` | 組版のためのスクリプト |

## この本の作り方について

本書の本文・コード・図は、AI（Anthropic の Claude）を使って執筆し、編集しました。
本文に載せた実行結果と数値は、このリポジトリのコードを実際に動かして得たものです。
ただし、専門家による査読は受けていません。誤りが残っている可能性があります。

分かっている問題:

- 付録 B の手順は Linux で確かめました。macOS と Windows でも動く書き方にしてありますが、Windows での実行は確かめていません。
- 第 12 章の `torch.compile` の効果は実測していません（本文にもそう書いてあります）。
- 組版すると、行がわずかにはみ出す箇所が 24 か所あります（どれも 10 pt 未満）。

誤りや分かりにくいところを見つけたら、[Issues](https://github.com/k-tamuraphys/zero-kara-llm/issues) で知らせてください。

## 名前について

本書は独立した著作物です。書名の似たほかの書籍（オライリー・ジャパンの『ゼロから作る Deep Learning』シリーズなど）や、その著者・出版社とは関係がありません。

## ライセンス

- **コード**（`code/`、`tools/`）: [MIT License](LICENSE-CODE)
- **本文**（`chapters/`、`main.tex`、`preamble.tex`、`refs.bib`、図、PDF）: [CC BY-NC-SA 4.0](LICENSE-TEXT)

くわしくは [`LICENSE.md`](LICENSE.md) を見てください。
