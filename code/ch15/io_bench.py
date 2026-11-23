"""小さなファイル多数 vs 連結した 1 ファイル：読み出しの速さを測る。

同じ 16 MiB のデータを (a) 8 KiB のファイル 2,000 個 (b) 1 個のファイル
として置き、全部読む時間を比べる。置き場所はローカルディスクと NFS。
python ch15/io_bench.py   （一時ファイルは最後に消す）
"""
import csv
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
N_FILES, SIZE, REPEAT = 2000, 8192, 5


def fs_type(path):
    """置き場所のファイルシステムの種類を /proc/mounts から調べる。"""
    best, kind = "", "?"
    for line in open("/proc/mounts"):
        _, mnt, typ = line.split()[:3]
        if str(path.resolve()).startswith(mnt) and len(mnt) > len(best):
            best, kind = mnt, typ
    return kind


def sync(path):
    """書いた内容をディスク（またはサーバ）まで確実に届ける。"""
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def evict(path):
    """OS に「このファイルのキャッシュは捨ててよい」と伝える。"""
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


# [snippet: readers]
def read_small(d, order):
    """小さなファイルを 1 個ずつ開いて読む。"""
    total = 0
    for i in order:
        with open(d / f"{i:05d}.bin", "rb") as f:
            total += len(f.read())
    return total


def read_memmap(d, order):
    """連結した 1 ファイルを memmap し、同じ順に切り出す。"""
    m = np.memmap(d / "all.bin", dtype=np.uint8, mode="r")
    total = 0
    for i in order:
        total += len(bytes(m[i * SIZE:(i + 1) * SIZE]))
    return total
# [end: readers]


def read_one(d, order):
    """連結した 1 ファイルを開き、同じ順に 8 KiB ずつ読む。"""
    total = 0
    with open(d / "all.bin", "rb") as f:
        for i in order:
            f.seek(i * SIZE)
            total += len(f.read(SIZE))
    return total


def bench(root, rng):
    d = Path(tempfile.mkdtemp(prefix="ch15_io_", dir=root))
    try:
        blob = rng.integers(0, 256, N_FILES * SIZE, dtype=np.uint8)
        t0 = time.perf_counter()
        for i in range(N_FILES):
            with open(d / f"{i:05d}.bin", "wb") as f:
                f.write(blob[i * SIZE:(i + 1) * SIZE].tobytes())
        t_small = time.perf_counter() - t0
        t0 = time.perf_counter()
        with open(d / "all.bin", "wb") as f:
            f.write(blob.tobytes())
            f.flush()
            os.fsync(f.fileno())
        t_one = time.perf_counter() - t0
        res = {"write_small": t_small, "write_one": t_one}
        for p in d.iterdir():
            sync(p)
        for name, fn in (("small", read_small), ("one", read_one),
                         ("memmap", read_memmap)):
            times = []
            for _ in range(REPEAT):
                for p in d.iterdir():       # 毎回キャッシュを追い出す
                    evict(p)
                order = rng.permutation(N_FILES)
                t0 = time.perf_counter()
                assert fn(d, order) == N_FILES * SIZE
                times.append(time.perf_counter() - t0)
            res[name] = statistics.median(times)
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    places = [("local", "ローカルディスク", Path(tempfile.gettempdir())),
              ("shared", "ネットワーク越し", OUT)]
    mb = N_FILES * SIZE / 1e6
    lines = [f"データ: {SIZE // 1024} KiB × {N_FILES:,} 個 = {mb:.1f} MB"
             f"（読み出しは {REPEAT} 回の中央値）"]
    rows = []
    for key, label, root in places:
        kind = fs_type(root)
        r = bench(root, rng)
        lines.append(f"== 置き場所: {label}（{kind}）==")
        lines.append(f"書き込み: 小さなファイル {r['write_small']:.2f} 秒"
                     f" / 1 ファイル {r['write_one']:.2f} 秒")
        for name, what in (("small", f"小さなファイル {N_FILES:,} 個"),
                           ("one", "1 ファイル seek+read  "),
                           ("memmap", "1 ファイル memmap     ")):
            t = r[name]
            lines.append(f"{what}: {t * 1e3:8.1f} ミリ秒（1 個 "
                         f"{t / N_FILES * 1e6:7.1f} マイクロ秒）")
            rows.append((key, kind, name, f"{t * 1e3:.2f}",
                         f"{t / N_FILES * 1e6:.2f}"))
        lines.append(f"→ 小さなファイルは seek+read の "
                     f"{r['small'] / r['one']:.1f} 倍、memmap の "
                     f"{r['small'] / r['memmap']:.1f} 倍の時間")
    (OUT / "io_bench.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    with open(OUT / "io_bench.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place", "fs", "method", "ms", "us_per_file"])
        w.writerows(rows)
    with open(OUT / "io_bench_wide.csv", "w", newline="") as f:
        w = csv.writer(f)                   # 図のための横持ちの表
        w.writerow(["idx", "method", "local_ms", "shared_ms"])
        half = len(rows) // 2
        for i in range(half):
            w.writerow([i, rows[i][2], rows[i][3], rows[half + i][3]])
