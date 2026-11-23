"""1 回の生成にかかる時間の内訳: 最初のトークンまで（TTFT）と、その後。"""

import csv
import time

import numpy as np
import torch

from common import load
from engine import Engine, pick
from llmbook.corpus import DATA_DIR

model, tok = load("base")  # 長い文章を書かせたいので、基盤モデルを使う
torch.set_num_threads(1)  # 時間の測定を安定させるため、1 スレッドで動かす
val = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16).astype(np.int64)
N_PROMPT, N_NEW, REPEAT = 16, 100, 8


def best(times: list[np.ndarray]) -> np.ndarray:
    """何回か測ったうちの最小値（ほかの処理に邪魔されなかった回）を使う。"""
    return np.min(np.stack(times), axis=0)


# [snippet: timed-generate]
@torch.no_grad()
def timed_plain(prompt: list[int], n_new: int) -> np.ndarray:
    """キャッシュなし: 毎回、列全体を計算し直す。各トークンの時間を返す。"""
    idx, times = torch.tensor([prompt]), []
    for _ in range(n_new):
        t0 = time.perf_counter()
        logits, _ = model(idx)                    # 列全体を計算する
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        times.append(time.perf_counter() - t0)
    return np.array(times)


@torch.no_grad()
def timed_cached(prompt: list[int], n_new: int) -> np.ndarray:
    """キャッシュあり: プロンプトを 1 回読ませ、あとは 1 トークンずつ足す。"""
    engine, slot, times = Engine(model, n_slots=1), torch.tensor([0]), []
    t0 = time.perf_counter()
    nxt = pick(engine.forward(slot, torch.tensor([prompt])))  # プロンプト全体
    times.append(time.perf_counter() - t0)
    for _ in range(n_new - 1):
        t0 = time.perf_counter()
        nxt = pick(engine.forward(slot, nxt[:, None]))        # 新しい 1 個だけ
        times.append(time.perf_counter() - t0)
    return np.array(times)
# [end: timed-generate]


if __name__ == "__main__":
    prompt = val[1000:1000 + N_PROMPT].tolist()
    runs_plain, runs_cached = [], []
    for _ in range(REPEAT):  # 二つを交互に測る（混雑の影響を公平にする）
        runs_plain.append(timed_plain(prompt, N_NEW))
        runs_cached.append(timed_cached(prompt, N_NEW))
    plain, cached = best(runs_plain), best(runs_cached)
    print(f"プロンプト {N_PROMPT} トークン → {N_NEW} トークン生成（貪欲法）")
    for name, t in [("キャッシュなし", plain), ("キャッシュあり", cached)]:
        print(f"  {name}: 合計 {t.sum() * 1000:6.1f} ミリ秒"
              f"  = {N_NEW / t.sum():6.0f} トークン/秒"
              f"  | 1 個目 {t[0] * 1000:5.2f}  10 個目 {t[9] * 1000:5.2f}"
              f"  100 個目 {t[99] * 1000:5.2f} ミリ秒")
    print(f"  速度の比: {plain.sum() / cached.sum():.1f} 倍")
    with open("ch19/out/per_token.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "plain_ms", "cached_ms"])
        for i in range(0, N_NEW, 3):
            w.writerow([i + 1, f"{plain[i] * 1000:.3f}",
                        f"{cached[i] * 1000:.3f}"])

    # プロンプトの長さと、最初のトークンまでの時間（TTFT）
    print("\nプロンプトの長さと TTFT（キャッシュあり）")
    with open("ch19/out/ttft.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_prompt", "ttft_ms", "next_ms"])
        for n in [4, 8, 16, 32, 64, 96, 120]:
            p = val[1000:1000 + n].tolist()
            t = best([timed_cached(p, 6) for _ in range(REPEAT * 2)])
            w.writerow([n, f"{t[0] * 1000:.3f}", f"{t[1:].mean() * 1000:.3f}"])
            print(f"  {n:3d} トークン: 1 個目まで {t[0] * 1000:5.2f} ミリ秒"
                  f"   2 個目以降 1 個あたり {t[1:].mean() * 1000:5.2f} ミリ秒"
                  )
