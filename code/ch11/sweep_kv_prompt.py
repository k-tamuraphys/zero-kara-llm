"""プロンプト（前置き）の長さを変えて、KV キャッシュの効き方を測る。

kv_cache.py と同じ学習前のモデルで、新しいトークンを 64 個生成する時間を、
(1) キャッシュなし、(2) キャッシュあり、(3) 前置きのキャッシュを使い回す
（前置きの計算を済ませておき、生成の部分だけを測る）の 3 通りで比べる。
使い方: python ch11/sweep_kv_prompt.py > ch11/out/sweep_kv_prompt.txt
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, "ch11")
from modern_gpt import KVCache, ModernConfig, ModernGPT

torch.set_num_threads(2)
torch.manual_seed(0)
cfg = ModernConfig(n_layer=4, n_head=4, n_kv_head=4, n_embd=192,
                   ffn_hidden=512, block_size=128)   # kv_cache.py と同じ
model = ModernGPT(cfg).eval()
n_new = 64


@torch.no_grad()
def gen_reuse(saved, last_logits, T0):
    """保存しておいた前置きのキャッシュから、生成の部分だけを行う。"""
    caches = []
    for k, v in saved:                    # 箱だけ新しくし、中身は共有する
        c = KVCache()
        c.k, c.v = k, v                   # update は cat で新しく作るので安全
        caches.append(c)
    logits = last_logits
    for t in range(n_new):
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        logits, _ = model(nxt, caches=caches, start=T0 + t)


def median_ms(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


rows = []
for T0 in [16, 64, 128, 256, 448]:
    prompt = torch.randint(0, cfg.vocab_size, (1, T0))
    t_no = median_ms(lambda: model.generate(prompt, n_new, use_cache=False))
    t_yes = median_ms(lambda: model.generate(prompt, n_new, use_cache=True))
    with torch.no_grad():                 # 前置きを一度だけ計算して保存
        caches = [KVCache() for _ in model.blocks]
        logits, _ = model(prompt, caches=caches)
    saved = [(c.k, c.v) for c in caches]
    t_re = median_ms(lambda: gen_reuse(saved, logits, T0))
    rows.append((T0, t_no, t_yes, t_re))
    print(f"前置き {T0:3d} トークン: なし {t_no:7.1f} ms  あり {t_yes:6.1f} ms"
          f"  使い回し {t_re:6.1f} ms")
with open("ch11/out/sweep_kv_prompt.csv", "w") as f:
    f.write("T0,nocache,cache,reuse\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]:.1f},{r[2]:.1f},{r[3]:.1f}\n")
