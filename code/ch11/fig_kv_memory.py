"""公開モデルの設定から、1 トークンあたりの KV キャッシュを計算する（図用）。

式 (11.7): 2 × L × H_kv × d_head × b バイト。設定の値は
research/models_and_methods.md §1.1・§1.2 による。線形注意との混成モデルは、
文脈の長さに比例してキャッシュが増える「完全な注意」の層だけを L に数える。
使い方: python ch11/fig_kv_memory.py > ch11/out/fig_kv_memory.txt
"""


def kv_bytes_per_token(L, Hkv, dh, bytes_per):          # param_count.py と同じ
    return 2 * L * Hkv * dh * bytes_per                 # K と V の 2 つ

models = [
    # 名前, キャッシュを持つ層の数 L, H_kv, d_head, 1 つの数のバイト数
    ("ミニGPT（第10章・FP32）", 4, 4, 48, 4),
    ("現代版ミニGPT（FP32）", 4, 2, 32, 4),
    ("GPT-2 124M", 12, 12, 64, 2),
    ("Llama 2 7B", 32, 32, 128, 2),
    ("Llama 3.1 8B", 32, 8, 128, 2),
    ("Qwen3-8B", 36, 8, 128, 2),
    ("Qwen3-30B-A3B", 48, 4, 128, 2),
    ("gpt-oss-120b", 36, 8, 64, 2),
    ("Llama 3.1 405B", 126, 8, 128, 2),
    ("Qwen3.5-9B（32層中8層）", 8, 4, 256, 2),
    ("Qwen3.8-27B（64層中16層）", 16, 4, 256, 2),
]
ctx = 131072                                    # 文脈 131,072 トークン
with open("ch11/out/fig_kv_memory.csv", "w") as f:
    f.write("name,kib,gib_ctx\n")
    for name, L, hkv, dh, b in models:
        kv = kv_bytes_per_token(L, hkv, dh, b)
        f.write(f"{name},{kv / 1024:.0f},{kv * ctx / 2**30:.2f}\n")
        print(f"{name:24s} {kv:8,d} バイト = {kv / 1024:4.0f} KiB  "
              f"{ctx:,} トークンで {kv * ctx / 2**30:6.2f} GiB")
