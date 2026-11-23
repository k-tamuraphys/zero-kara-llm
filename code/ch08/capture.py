"""print の出力を画面に出しつつためておき、out/ のファイルに保存する補助。"""

import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"


class Capture:
    def __init__(self):
        self.buf: list[str] = []
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.buf.append(s)

    def flush(self) -> None:
        self.stdout.flush()

    def save(self, name: str) -> None:
        """ここまでにためた出力を out/name に書き、ためた分を捨てる。"""
        OUT.mkdir(exist_ok=True)
        (OUT / name).write_text("".join(self.buf), encoding="utf-8")
        self.buf.clear()
