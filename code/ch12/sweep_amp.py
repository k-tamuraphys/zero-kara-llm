"""BF16 の混合精度と FP32 で、毎ステップの損失を記録して比べる。

amp_train.py と同じ設定（ミニGPT、B=32, T=128、AdamW lr=1e-3、61 ステップ）。
乱数の種（初期値とバッチの選び方）を 0 と 1 の 2 通りに変え、
「BF16 と FP32 の差」を「種を変えたときの差」と比べられるようにする。
"""
import time
import numpy as np
import torch
from llmbook.model import GPT, GPTConfig
from llmbook.train import get_batch

torch.set_num_threads(2)            # amp_train.py と同じ 2 スレッド
data = torch.from_numpy(np.fromfile("data/train.bin", dtype=np.uint16)
                        .astype(np.int64))


def run(mode, seed, steps=61):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4,
                    n_embd=192)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        x, y = get_batch(data, 32, 128)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16,
                            enabled=(mode == "bf16")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


curves = {}
for mode in ("fp32", "bf16"):
    for seed in (0, 1):
        t0 = time.perf_counter()
        curves[f"{mode}_s{seed}"] = run(mode, seed)
        c = curves[f"{mode}_s{seed}"]
        print(f"{mode} 種 {seed}: 損失 0 歩目 {c[0]:.4f}, 20 歩目 {c[20]:.4f},"
              f" 60 歩目 {c[60]:.4f}（{time.perf_counter() - t0:.0f} 秒）")

names = list(curves)
with open("ch12/out/sweep_amp.csv", "w") as f:
    f.write("step," + ",".join(names) + "\n")
    for i in range(61):
        f.write(f"{i}," + ",".join(f"{curves[n][i]:.4f}" for n in names)
                + "\n")
# 差の大きさ: 同じ種での BF16 と FP32 の差、同じ形式で種を変えた差
for a, b in (("bf16_s0", "fp32_s0"), ("bf16_s1", "fp32_s1"),
             ("fp32_s1", "fp32_s0"), ("bf16_s1", "bf16_s0")):
    d = [abs(x - y) for x, y in zip(curves[a][10:], curves[b][10:])]
    print(f"|{a} - {b}|: 10〜60 歩目の平均 {sum(d) / len(d):.3f},"
          f" 最大 {max(d):.3f}")
