"""学習時のメモリの内訳: 式による見積もりと、活性化の実測。"""
import torch
from llmbook.model import GPT, GPTConfig

torch.manual_seed(0)
torch.set_num_threads(2)


# [snippet: states]
def state_bytes(N, mixed=True):
    """パラメータ数 N の学習で、活性化以外に常に必要なバイト数（AdamW）。"""
    if mixed:   # bf16 の重みと勾配 + fp32 の主重み + fp32 の Adam の m, v
        return {"重み": 2 * N, "勾配": 2 * N, "主重み": 4 * N,
                "Adam": 8 * N}
    return {"重み": 4 * N, "勾配": 4 * N, "Adam": 8 * N}  # 全部 fp32
# [end: states]


# [snippet: activations]
def activation_bytes(model, B, T):
    """逆伝播のために保存されるテンソルの合計バイト数を数える。"""
    total = 0
    params = {q.data_ptr() for q in model.parameters()}
    def pack(t):              # 順伝播で何かが保存されるたびに呼ばれる
        nonlocal total
        if t.data_ptr() not in params:   # 重みそのものは数えない
            total += t.numel() * t.element_size()
        return t
    x = torch.randint(0, model.config.vocab_size, (B, T))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = model(x, x)
    return total
# [end: activations]


cfg = GPTConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4,
                n_embd=192)
model = GPT(cfg)
N = model.num_params()
s32 = sum(state_bytes(N, False).values())
print(f"ミニGPT N={N:,}: 状態（fp32） {s32 / 2**20:.1f} MiB")
rows = []
for B in (1, 2, 4, 8, 16, 32):
    for T in (32, 64, 128):
        a = activation_bytes(model, B, T)
        rows.append((B, T, a))
        if T == 128 or B == 32:
            print(f"B={B:2d} T={T:3d}: 活性化 {a / 2**20:7.1f} MiB"
                  f" ({a / (B * T) / 1024:.1f} KiB/トークン)")
with open("ch12/out/activation.csv", "w") as f:
    f.write("B,T,mib\n")
    for B, T, a in rows:
        f.write(f"{B},{T},{a / 2**20:.2f}\n")
with open("ch12/out/activation_T128.csv", "w") as f:   # 図用
    f.write("B,mib\n")
    for B, T, a in rows:
        if T == 128:
            f.write(f"{B},{a / 2**20:.2f}\n")

print("== 公開モデル規模の見積もり（混合精度 + AdamW, 活性化を除く）==")
with open("ch12/out/memory_models.csv", "w") as f, \
        open("ch12/out/memory_models_small.csv", "w") as g:
    f.write("name,N,weights,grads,master,adam,total\n")
    g.write("name,weights,grads,master,adam\n")
    for name, n in [("GPT-2 124M", 124e6), ("1B", 1e9),
                    ("Llama 3 8B", 8.03e9), ("Llama 3 70B", 70.55e9)]:
        s = state_bytes(n)
        tot = sum(s.values())
        print(f"{name:12s}: " + ", ".join(f"{k} {v / 1e9:.1f}" for k, v in s.items())
              + f" -> 合計 {tot / 1e9:.1f} GB")
        vals = ",".join(f"{v / 1e9:.2f}" for v in s.values())
        f.write(f"{name},{n:.3g},{vals},{tot / 1e9:.2f}\n")
        if "70B" not in name:                             # 図用
            g.write(f"{name},{vals}\n")
