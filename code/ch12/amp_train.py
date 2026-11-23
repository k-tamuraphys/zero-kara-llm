"""混合精度（bf16 autocast）の学習ループ。GPU があれば GPU で、無ければ CPU で動く。"""
import sys
import time
import numpy as np
import torch
from llmbook.model import GPT, GPTConfig
from llmbook.train import get_batch

torch.set_num_threads(2)
mode = sys.argv[1] if len(sys.argv) > 1 else "bf16"     # "fp32" か "bf16"

# [snippet: device]
device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.from_numpy(np.fromfile("data/train.bin", dtype=np.uint16)
                        .astype(np.int64))
torch.manual_seed(0)
cfg = GPTConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4,
                n_embd=192)
model = GPT(cfg).to(device)              # 重みを GPU のメモリへ移す
opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                        fused=(device == "cuda"))   # GPU では融合版
# [end: device]
if device == "cuda":
    # [snippet: gpu-only]
    torch.set_float32_matmul_precision("high")   # fp32 の行列積に TF32 を許す
    model = torch.compile(model)                 # 演算を融合して速くする
    # [end: gpu-only]

# [snippet: amp-loop]
use_bf16 = mode == "bf16"
t0 = time.perf_counter()
for step in range(61):
    x, y = get_batch(data, 32, 128, device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16,
                        enabled=use_bf16):
        logits, loss = model(x, y)        # 行列積は bf16 で計算される
    opt.zero_grad(set_to_none=True)
    loss.backward()                       # 勾配は fp32 の重みに対して溜まる
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()                            # 更新は fp32 の主重みで行う
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  logits {logits.dtype}"
              f"  重み {model.lm_head.weight.dtype}")
# [end: amp-loop]
if device == "cuda":
    torch.cuda.synchronize()
print(f"{mode}: 61 ステップ {time.perf_counter() - t0:.1f} 秒 (device={device})")
