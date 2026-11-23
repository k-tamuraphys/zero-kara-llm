"""青空文庫から夏目漱石の作品を取得し、学習用のテキストに整形する。

青空文庫のテキストファイルは Shift_JIS で書かれ、ルビ（ふりがな）や
入力者注などの記号を含む。言語モデルの学習にはこれらが邪魔になるので、
本文だけを取り出して 1 つの UTF-8 テキストにまとめる。
"""

import io
import re
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 作品名と、青空文庫のテキストファイル（zip）の場所。すべて新字新仮名版。
_BASE = "https://www.aozora.gr.jp/cards/000148/files/"
WORKS = {
    "吾輩は猫である": _BASE + "789_ruby_5639.zip",
    "坊っちゃん": _BASE + "752_ruby_2438.zip",
    "草枕": _BASE + "776_ruby_6020.zip",
    "虞美人草": _BASE + "761_ruby_1861.zip",
    "坑夫": _BASE + "774_ruby_1640.zip",
    "三四郎": _BASE + "794_ruby_4237.zip",
    "それから": _BASE + "56143_ruby_50824.zip",
    "門": _BASE + "785_ruby_1656.zip",
    "彼岸過迄": _BASE + "765_ruby_2469.zip",
    "行人": _BASE + "775_ruby_2064.zip",
    "こころ": _BASE + "773_ruby_5968.zip",
    "道草": _BASE + "783_ruby_1311.zip",
    "明暗": _BASE + "782_ruby_2315.zip",
    "夢十夜": _BASE + "799_ruby_6024.zip",
}


def download_work(url: str) -> str:
    """zip を取得し、中のテキストファイルを文字列として返す。"""
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = next(n for n in archive.namelist() if n.endswith(".txt"))
    return archive.read(name).decode("cp932")  # cp932 は Shift_JIS の拡張


def clean_aozora(raw: str) -> str:
    """青空文庫形式のテキストから本文だけを取り出す。"""
    text = raw.replace("\r\n", "\n")
    # 冒頭: 題名・著者名のあと、点線で囲まれた記号の説明がある。
    # 2 本目の点線より後ろが本文である。
    parts = re.split(r"^-{10,}\n", text, flags=re.MULTILINE)
    if len(parts) >= 3:
        text = parts[2]
    # 末尾: 「底本：」以降は書誌情報なので捨てる。
    text = re.split(r"^底本[：:]", text, flags=re.MULTILINE)[0]
    text = re.sub(r"※?［＃[^］]*］", "", text)  # 入力者注 ［＃...］
    text = re.sub(r"《[^》]*》", "", text)  # ルビ 《...》
    text = text.replace("｜", "")  # ルビの開始位置を示す縦棒
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3 行以上の空行は 1 つに
    return text.strip() + "\n"


def build_corpus(force: bool = False) -> Path:
    """全作品を取得・整形し data/soseki.txt に保存する。"""
    out_path = DATA_DIR / "soseki.txt"
    if out_path.exists() and not force:
        return out_path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = []
    for title, url in WORKS.items():
        body = clean_aozora(download_work(url))
        print(f"{title}: {len(body):,} 文字")
        texts.append(body)
    out_path.write_text("\n".join(texts), encoding="utf-8")
    return out_path


def load_corpus() -> str:
    """整形済みコーパスを文字列として読み込む（なければ取得する）。"""
    return build_corpus().read_text(encoding="utf-8")


def train_val_split(text: str, val_ratio: float = 0.05) -> tuple[str, str]:
    """先頭を訓練用、末尾を検証用に分ける。"""
    n_val = int(len(text) * val_ratio)
    return text[:-n_val], text[-n_val:]


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"合計: {len(corpus):,} 文字, 異なり文字数: {len(set(corpus)):,}")
