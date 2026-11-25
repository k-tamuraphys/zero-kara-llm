"""MinHash のハッシュ関数の数を変えて、Jaccard 係数の推定のぶれを測る。

実行: python ch13/sweep_minhash.py  （約 10 秒）
出力: ch13/out/sweep_minhash.txt, ch13/out/sweep_minhash.csv
"""

import hashlib
import random
import statistics
import zlib

# data_pipeline.py の A と B（B は A から「ただ」を消した転載）
A = ("私はその人を常に先生と呼んでいた。だからここでもただ先生と書く"
     "だけで本名は打ち明けない。")
B = ("私はその人を常に先生と呼んでいた。だからここでも先生と書くだけ"
     "で本名は打ち明けない。")


def shingles(text: str, n: int = 5) -> set[str]:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def crc(text: str) -> int:
    return zlib.crc32(text.encode())  # data_pipeline.py と同じ関数


def blake(text: str) -> int:
    # 比較用: 暗号にも使われる、よりでたらめに近いハッシュ関数
    return int.from_bytes(hashlib.blake2b(text.encode(),
                                          digest_size=8).digest(), "big")


def minhash_est(a: set, b: set, n_hash: int, seed: int, h) -> float:
    """data_pipeline.py の minhash と同じ作り方で、一致の割合を返す。"""
    rng = random.Random(seed)
    salts = [rng.getrandbits(32) for _ in range(n_hash)]
    same = 0
    for salt in salts:
        ha = min(h(f"{salt}:{x}") for x in a)
        hb = min(h(f"{salt}:{x}") for x in b)
        same += ha == hb
    return same / n_hash


sa, sb = shingles(A), shingles(B)
J = len(sa & sb) / len(sa | sb)
N_SEEDS = 50  # 塩の選び方（乱数の種）を 50 通り変える
print(f"正しい Jaccard 係数: {J:.3f}")
with open("ch13/out/sweep_minhash.csv", "w") as f:
    f.write("hash,k,mean,lo,hi,sd,se\n")
    for name, h in [("crc32", crc), ("blake2b", blake)]:
        for k in [8, 16, 32, 64, 128, 256, 512]:
            est = [minhash_est(sa, sb, k, seed, h)
                   for seed in range(N_SEEDS)]
            mean, sd = statistics.mean(est), statistics.pstdev(est)
            se = (J * (1 - J) / k) ** 0.5  # 理論上の目安 √(J(1-J)/k)
            f.write(f"{name},{k},{mean:.4f},{min(est):.4f},"
                    f"{max(est):.4f},{sd:.4f},{se:.4f}\n")
            print(f"{name:7s} k={k:3d}: 平均 {mean:.3f} / 最小 "
                  f"{min(est):.3f} / 最大 {max(est):.3f} / 標準偏差 "
                  f"{sd:.3f}（目安 {se:.3f}）")
