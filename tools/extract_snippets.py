#!/usr/bin/env python3
"""code/ 以下のソースから、本文に掲載するスニペットを build/snippets/ に切り出す。

ソース中の目印:
    # [snippet: 名前]
    ...掲載したい行...
    # [end: 名前]

出力:
    build/snippets/<code からの相対パス>/<名前>.txt   … 目印で囲んだ部分
    build/snippets/<code からの相対パス>/_full.txt    … ファイル全体（目印の行は除く）

スニペットは入れ子・重なりを許す。共通の字下げは取り除く。
"""

import os
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUT = ROOT / "build" / "snippets"
START = re.compile(r"^\s*#\s*\[snippet:\s*([\w\-]+)\s*\]\s*$")
END = re.compile(r"^\s*#\s*\[end:\s*([\w\-]+)\s*\]\s*$")
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".txt", ".slurm", ".md"}
MAX_COLS = 79


def width(line: str) -> int:
    """全角文字を 2 桁として数えた表示幅。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in line)


def write_if_changed(path: Path, body: str) -> None:
    """内容が変わったときだけ、一時ファイル経由で原子的に書き換える。

    複数の章を同時に組版しても、読みかけのファイルが空になることがない。
    """
    if path.exists() and path.read_text(encoding="utf-8") == body:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def process(path: Path) -> list[str]:
    problems = []
    rel = path.relative_to(CODE)
    lines = path.read_text(encoding="utf-8").split("\n")
    open_snips: dict[str, list[str]] = {}
    done: dict[str, list[str]] = {}
    full = []
    for no, line in enumerate(lines, 1):
        m = START.match(line)
        if m:
            open_snips[m.group(1)] = []
            continue
        m = END.match(line)
        if m:
            name = m.group(1)
            if name not in open_snips:
                problems.append(f"{rel}:{no}: [end: {name}] に対応する開始がない")
            else:
                done[name] = open_snips.pop(name)
            continue
        full.append(line)
        for buf in open_snips.values():
            buf.append(line)
        if path.suffix == ".py" and width(line) > MAX_COLS:
            problems.append(f"{rel}:{no}: 行が長すぎる（{width(line)} 桁 > {MAX_COLS}）")
    for name in open_snips:
        problems.append(f"{rel}: [snippet: {name}] が閉じられていない")
    out_dir = OUT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    write_if_changed(out_dir / "_full.txt", "\n".join(full).strip("\n") + "\n")
    for name, buf in done.items():
        body = textwrap.dedent("\n".join(buf)).strip("\n") + "\n"
        write_if_changed(out_dir / f"{name}.txt", body)
    return problems


def main() -> int:
    problems = []
    for path in sorted(CODE.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(part in {"data", "checkpoints", "__pycache__", "out"} for part in path.parts):
            continue
        problems += process(path)
    for p in problems:
        print("WARNING:", p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
