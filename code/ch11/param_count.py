"""公開モデルの設定から、パラメータ数と KV キャッシュの大きさを計算する。"""

# [snippet: count]
def count_params(V, d, L, H, Hkv, dh, ff, tied, n_exp=1, top_k=1,
                 shared=0):
    attn = d * H * dh + 2 * d * Hkv * dh + H * dh * d   # Wq, Wk, Wv, Wo
    ffn_one = 3 * d * ff                                # SwiGLU は行列 3 枚
    router = d * n_exp if n_exp > 1 else 0
    norms = 2 * d
    emb = V * d * (1 if tied else 2)                    # 入力と出力
    total = L * (attn + (n_exp + shared) * ffn_one + router + norms) + emb + d
    active = L * (attn + (top_k + shared) * ffn_one + router + norms) + emb + d
    return total, active


def kv_bytes_per_token(L, Hkv, dh, bytes_per=2):        # bf16 は 2 バイト
    return 2 * L * Hkv * dh * bytes_per                 # K と V の 2 つ
# [end: count]


models = {
    # 名前: (V, d, L, H, Hkv, dh, ff, tied, n_exp, top_k, shared)
    "Llama 2 7B": (32000, 4096, 32, 32, 32, 128, 11008, False, 1, 1, 0),
    "Llama 3 8B": (128256, 4096, 32, 32, 8, 128, 14336, False, 1, 1, 0),
    "Llama 3 70B": (128256, 8192, 80, 64, 8, 128, 28672, False, 1, 1, 0),
    "Mistral 7B": (32000, 4096, 32, 32, 8, 128, 14336, False, 1, 1, 0),
    "Mixtral 8x7B": (32000, 4096, 32, 32, 8, 128, 14336, False, 8, 2, 0),
    "Qwen3-8B": (151936, 4096, 36, 32, 8, 128, 12288, False, 1, 1, 0),
    "Qwen3-30B-A3B": (151936, 2048, 48, 32, 4, 128, 768, False, 128, 8, 0),
}
print("model             total    active   KV/token  KV(8192 tok)")
for name, c in models.items():
    V, d, L, H, Hkv, dh, ff, tied, E, k, s = c
    tot, act = count_params(V, d, L, H, Hkv, dh, ff, tied, E, k, s)
    kv = kv_bytes_per_token(L, Hkv, dh)
    print(f"{name:16s} {tot / 1e9:8.2f}B {act / 1e9:8.2f}B "
          f"{kv / 1024:5.0f} KiB {kv * 8192 / 2**30:8.2f} GiB")
