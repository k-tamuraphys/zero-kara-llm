"""チェックポイントの大きさと書き込み速度、データ供給の速さを測る。

(1) ミニGPT の完全なチェックポイントは本当に 12 バイト/パラメータか
(2) 256 MB のテンソルをローカルディスクと NFS に書く速さ
(3) get_batch がトークンを供給する速さと、学習が消費する速さ
python ch15/write_bench.py   （一時ファイルは最後に消す）
"""
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ch15.estimate import ckpt_bytes, hms, tokens_per_s, write_seconds
from ch15.io_bench import fs_type
from llmbook.corpus import DATA_DIR
from llmbook.model import GPT, GPTConfig
from llmbook.train import get_batch, make_optimizer

torch.set_num_threads(2)
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
lines = []

# ---- (1) 完全なチェックポイントの大きさ ----
model = GPT(GPTConfig(vocab_size=4096, block_size=128, n_layer=4,
                      n_head=4, n_embd=192))
opt = make_optimizer(model, lr=1e-3)
data = torch.from_numpy(np.fromfile(DATA_DIR / "train.bin",
                                    dtype=np.uint16).astype(np.int64))
x, y = get_batch(data, 32, 128)
step_times = []
for _ in range(5):                      # 学習の消費速度も同時に測る
    t0 = time.perf_counter()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    step_times.append(time.perf_counter() - t0)
n = model.num_params()
tmp = Path(tempfile.mkdtemp(prefix="ch15_ck_"))
# [snippet: full-ckpt]
state = {"model": model.state_dict(), "opt": opt.state_dict(),
         "step": 5, "rng": torch.get_rng_state()}
torch.save(state, tmp / "full.pt")
size = os.path.getsize(tmp / "full.pt")
# [end: full-ckpt]
os.remove(tmp / "full.pt")
os.rmdir(tmp)
lines += ["== (1) ミニGPT の完全なチェックポイント ==",
          f"見積もり 12N = {ckpt_bytes(n):,} バイト / 実際 {size:,} バイト"
          f" … 見積もり/実際 = {ckpt_bytes(n) / size:.3f}"]


# ---- (2) 書き込み速度（256 MB、fsync まで含める、3 回の中央値）----
# [snippet: write-speed]
def write_speed(root, n_bytes=256 * 2**20, repeat=3):
    blob = {"w": torch.randn(n_bytes // 4)}       # FP32 で n_bytes
    path = Path(root) / "ch15_write_test.pt"
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            torch.save(blob, f)
            f.flush()
            os.fsync(f.fileno())                  # 届くまで待つ
        times.append(time.perf_counter() - t0)
        os.remove(path)
    return n_bytes / statistics.median(times) / 1e9   # GB/s
# [end: write-speed]


lines.append("== (2) 256 MB を書く速さと、同じ速さで書いた場合の時間 ==")
for label, root in (("ローカルディスク", tempfile.gettempdir()),
                    ("ネットワーク越し", OUT)):
    v = write_speed(root)
    t7, t70 = (write_seconds(ckpt_bytes(m), v) for m in (7e9, 70e9))
    lines.append(f"{label}（{fs_type(Path(root))}）: {v:.2f} GB/s → "
                 f"7B（84 GB）{hms(t7)}、70B（840 GB）{hms(t70)}")

(OUT / "write_bench.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- (3) データの供給と消費 ----
t0 = time.perf_counter()
for _ in range(200):
    get_batch(data, 32, 128)
supply = 200 * 32 * 128 / (time.perf_counter() - t0)
use = 32 * 128 / statistics.median(step_times)
lines = ["== データの供給と消費（ミニGPT、B=32、T=128）==",
          f"get_batch の供給: {supply:,.0f} トークン/秒",
          f"学習の消費      : {use:,.0f} トークン/秒"
          f"（1 ステップ {statistics.median(step_times) * 1e3:.0f} ミリ秒）",
          f"供給/消費 = {supply / use:.0f} 倍",
          "-- 大きな学習が消費する速さ（MFU 40%、4 バイト/トークン）--"]
for name, m, gpu, k in (("8B を H100 8 枚", 8e9, "H100", 8),
                        ("70B を H100 1,024 枚", 70e9, "H100", 1024),
                        ("405B を H100 16,000 枚", 405e9, "H100", 16000)):
    tps = tokens_per_s(m, gpu, k)
    lines.append(f"{name:<22}: {tps:>12,.0f} トークン/秒 = "
                 f"{tps * 4 / 1e6:.2f} MB/s")
(OUT / "read_rate.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
