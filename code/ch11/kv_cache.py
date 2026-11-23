"""KV キャッシュの有無で、生成の結果と速度を比べる。"""
import statistics
import sys
import time

import torch

sys.path.insert(0, "ch11")
from modern_gpt import KVCache, ModernConfig, ModernGPT

torch.set_num_threads(2)
torch.manual_seed(0)
cfg = ModernConfig(n_layer=4, n_head=4, n_kv_head=4, n_embd=192,
                   ffn_hidden=512, block_size=128)   # 学習前のモデルで十分
model = ModernGPT(cfg).eval()
prompt = torch.randint(0, cfg.vocab_size, (1, 16))

# [snippet: kv-check]
a = model.generate(prompt, 64, use_cache=False)
b = model.generate(prompt, 64, use_cache=True)
print("生成したトークン列は一致するか:", torch.equal(a, b))
with torch.no_grad():
    full, _ = model(a)                       # 全部を一度に計算した場合
    caches = [KVCache() for _ in model.blocks]
    step_logits = [model(a[:, :16], caches=caches)[0][:, -1]]
    for t in range(16, a.shape[1] - 1):      # 1 トークンずつ追加した場合
        out, _ = model(a[:, t:t + 1], caches=caches, start=t)
        step_logits.append(out[:, -1])
diff = (torch.stack(step_logits, 1) - full[:, 15:-1]).abs().max().item()
print(f"ロジットの最大の差: {diff:.2e}")
# [end: kv-check]

# [snippet: kv-bench]
def bench(n_new, use_cache, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        model.generate(prompt, n_new, use_cache=use_cache)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

rows = []
for n_new in [32, 64, 128, 256, 480]:
    t_no, t_yes = bench(n_new, False), bench(n_new, True)
    rows.append((n_new, t_no, t_yes))
    print(f"新トークン {n_new:3d}: なし {t_no * 1000:7.1f} ms  "
          f"あり {t_yes * 1000:6.1f} ms  {t_no / t_yes:4.1f} 倍")
# [end: kv-bench]
with open("ch11/out/kv_speed.csv", "w") as f:
    f.write("n,nocache,cache\n")
    for n, x, y in rows:
        f.write(f"{n},{x * 1000:.1f},{y * 1000:.1f}\n")
