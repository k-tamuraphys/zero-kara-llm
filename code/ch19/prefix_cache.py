"""プレフィックスキャッシュ: KV キャッシュを保存し、次の要求で使い回す。

会話の履歴（前置き）のキーとバリューをファイルに書き込み、次の要求では
それを読み出して、続きだけを計算する。時間と、出力の一致を確かめる。
"""

import csv
import os
import tempfile
import time

import numpy as np
import torch

from common import load
from engine import Engine, pick
from llmbook.chat import render_chat

model, tok = load("sft")
torch.set_num_threads(1)  # 時間の測定を安定させるため、1 スレッドで動かす
torch.manual_seed(0)
SLOT = torch.tensor([0])
# 保存先は 2 か所。手元のディスクの一時フォルダと、出力フォルダ
# （本書の測定環境では、出力フォルダは NFS でつないだ共有ディスクにある）
NEAR = os.path.join(tempfile.gettempdir(), "prefix_cache_kv.pt")
FAR = "ch19/out/prefix_cache_kv.pt"
REPEAT, N_NEW = 61, 4
END = tok.special_tokens["<|end|>"]
# 8 往復の会話（1 往復 14 トークン）と、新しい質問（10 トークン）
TURNS = [("12+35は？", "47です。"), ("58-19は？", "39です。"),
         ("24+61は？", "85です。"), ("90-27は？", "63です。"),
         ("33+48は？", "81です。"), ("76-52は？", "24です。"),
         ("15+69は？", "84です。"), ("87-43は？", "44です。")]
NEW_QUESTION = "38+47は？"


def history_ids(turns, question: str | None = None) -> list[int]:
    """会話の履歴（と、あれば新しい質問）を 1 本のトークン列にする。"""
    messages = []
    for q, a in turns:
        messages += [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}]
    if question is None:  # 履歴だけ: 最後の <|assistant|> を付けない
        return tok.encode(render_chat(messages))
    messages.append({"role": "user", "content": question})
    return tok.encode(render_chat(messages, add_generation_prompt=True))


# [snippet: prefix-cache]
def common_prefix(a: list[int], b: list[int]) -> int:
    """二つのトークン列が、先頭から何個まで一致するかを数える。"""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n

def write_cache(engine: Engine, ids: list[int], path: str) -> None:
    """書き込み: ids を読ませ、できたキーとバリューをファイルに保存する。"""
    engine.prefill(SLOT, [ids])
    n = len(ids)
    torch.save({"ids": ids, "K": engine.K[:, 0, :n].clone(),
                "V": engine.V[:, 0, :n].clone()}, path)

def read_cache(engine: Engine, cache: dict, ids: list[int]) -> torch.Tensor:
    """読み出し: 先頭の一致する部分を写し、続きだけを計算する。"""
    n = min(common_prefix(cache["ids"], ids), len(ids) - 1)
    engine.K[:, 0, :n] = cache["K"][:, :n]  # K: (層, スロット, 位置, ...)
    engine.V[:, 0, :n] = cache["V"][:, :n]
    engine.length[0] = n                    # ここまでは計算済みとする
    return engine.forward(SLOT, torch.tensor([ids[n:]]))
# [end: prefix-cache]


def timed(engine: Engine, first) -> tuple[float, float, torch.Tensor, list]:
    """first() で 1 個目を得て、N_NEW 個まで生成する。

    1 個目までの時間（TTFT）、2 個目以降の 1 個あたりの時間（ミリ秒）、
    1 個目のロジット、生成したトークンを返す。
    """
    t0 = time.perf_counter()
    logits = first()
    nxt = pick(logits)
    t1 = time.perf_counter()
    out = [int(nxt)]
    for _ in range(N_NEW - 1):
        nxt = pick(engine.forward(SLOT, nxt[:, None]))
        out.append(int(nxt))
    t2 = time.perf_counter()
    return ((t1 - t0) * 1000, (t2 - t1) * 1000 / (N_NEW - 1),
            logits.clone(), out)


def med(xs: list[float]) -> float:
    return float(np.median(xs))


WAYS = ["none", "mem", "near", "far"]


def measure(n_turns_list: list[int]) -> list[dict]:
    """前置きの長さを変えて、4 通りの方法で時間と出力を比べる。"""
    engine = Engine(model, n_slots=1)
    rows = []
    for k in n_turns_list:
        prefix = history_ids(TURNS[:k])
        ids = history_ids(TURNS[:k], NEW_QUESTION)
        assert ids[:len(prefix)] == prefix
        rows.append({"prefix": prefix, "ids": ids,
                     "t": {w: [] for w in WAYS + ["w_near", "w_far", "r"]},
                     "dec": {w: [] for w in WAYS}})
    for rep in range(REPEAT + 1):  # 交互に測る。最初の 1 回は捨てる
        for r in rows:
            prefix, ids = r["prefix"], r["ids"]
            t0 = time.perf_counter()
            write_cache(engine, prefix, NEAR)
            t1 = time.perf_counter()
            write_cache(engine, prefix, FAR)
            t2 = time.perf_counter()
            engine.prefill(SLOT, [prefix])  # 比較用: 保存せずに読むだけ
            t3 = time.perf_counter()
            cache = torch.load(NEAR)
            res = {
                "none": timed(engine, lambda: engine.prefill(SLOT, [ids])),
                "mem": timed(engine, lambda: read_cache(engine, cache, ids)),
                "near": timed(engine, lambda: read_cache(
                    engine, torch.load(NEAR), ids)),
                "far": timed(engine, lambda: read_cache(
                    engine, torch.load(FAR), ids)),
            }
            if rep == 0:
                continue
            for w, x in [("w_near", t1 - t0), ("w_far", t2 - t1),
                         ("r", t3 - t2)]:
                r["t"][w].append(x * 1000)
            for w in WAYS:
                r["t"][w].append(res[w][0])
                r["dec"][w].append(res[w][1])
            r["res"], r["bytes"] = res, os.path.getsize(NEAR)
            r["kv_bytes"] = 4 * (cache["K"].numel() + cache["V"].numel())
    return rows


def chat(use_cache: bool) -> tuple[int, float, str]:
    """8 回の質問を順に送る。毎回、履歴の全体を送り直す。

    use_cache なら、前の回のキーとバリューをスロットに残して使い回す。
    プリフィルしたトークン数の合計、その時間の合計、会話の全文を返す。
    """
    engine, held, messages = Engine(model, n_slots=1), [], []
    n_tok, ms = 0, 0.0
    for q, _ in TURNS:
        messages.append({"role": "user", "content": q})
        ids = tok.encode(render_chat(messages, add_generation_prompt=True))
        n = min(common_prefix(held, ids), len(ids) - 1) if use_cache else 0
        t0 = time.perf_counter()
        engine.length[0] = n
        nxt = pick(engine.forward(SLOT, torch.tensor([ids[n:]])))
        ms += (time.perf_counter() - t0) * 1000
        n_tok += len(ids) - n
        out = []
        while int(nxt) != END and len(out) < 5:
            out.append(int(nxt))
            nxt = pick(engine.forward(SLOT, nxt[:, None]))
        held = ids + out  # いまスロットにキーとバリューがあるトークン
        messages.append({"role": "assistant", "content": tok.decode(out)})
    return n_tok, ms, render_chat(messages)


if __name__ == "__main__":
    L, _, _, H, dh = Engine(model, n_slots=1).K.shape
    per_tok = 2 * L * H * dh * 4
    print(f"1 トークンの KV キャッシュ: 2 x {L} 層 x {H} ヘッド x {dh} 次元"
          f" x 4 バイト = {per_tok} バイト（{per_tok / 1024:.0f} KiB）")

    print("\n[1] 書き込みと読み出し（前置き + 新しい質問 10 トークン）")
    rows = measure([1, 2, 4, 6, 8])
    print("  保存したファイルの大きさ（バイト）")
    print("  前置き   ファイル   うち KV   KV / 前置きのトークン数")
    for r in rows:
        n = len(r["prefix"])
        print(f"  {n:5d}   {r['bytes']:8,d}  {r['kv_bytes']:8,d}"
              f"   {r['kv_bytes'] / n:.0f}")

    print("\n  1 個目までの時間（ミリ秒。中央値）")
    print("  前置き   なし   メモリ  ファイル（手元）  ファイル（共有）")
    for r in rows:
        t = {w: med(r["t"][w]) for w in WAYS}
        print(f"  {len(r['prefix']):5d}  {t['none']:5.2f}   {t['mem']:5.2f}"
              f"       {t['near']:5.2f}            {t['far']:5.2f}")
    print("\n  2 個目以降の 1 個あたりの時間（ミリ秒。中央値）")
    for r in rows:
        d = {w: med(r["dec"][w]) for w in WAYS}
        print(f"  {len(r['prefix']):5d}  {d['none']:5.2f}   {d['mem']:5.2f}"
              f"       {d['near']:5.2f}            {d['far']:5.2f}")

    print("\n  書き込みの時間（ミリ秒。中央値）")
    print("  前置き  読むだけ  読んで保存（手元）  読んで保存（共有）")
    for r in rows:
        print(f"  {len(r['prefix']):5d}   {med(r['t']['r']):5.2f}"
              f"         {med(r['t']['w_near']):5.2f}"
              f"              {med(r['t']['w_far']):5.2f}")

    print("\n  キャッシュなしの場合との、1 個目のロジットの差の最大値")
    print("  前置き  メモリ   手元    共有   生成した 4 トークン")
    engine4 = Engine(model, n_slots=4)
    for r in rows:
        ref, res = r["res"]["none"], r["res"]
        diff = [float((res[w][2] - ref[2]).abs().max())
                for w in ["mem", "near", "far"]]
        same = all(res[w][3] == ref[3] for w in WAYS)
        r["close"] = all(torch.allclose(res[w][2], ref[2], atol=1e-4)
                         for w in WAYS)
        print(f"  {len(r['prefix']):5d}  " + "  ".join(
            f"{d:.0e}" for d in diff) + f"   {'一致' if same else '不一致'}")
        # ほかの 3 件とまとめてプリフィルした場合（バッチの形が変わる）
        ids = r["ids"]
        both = engine4.prefill(torch.arange(4),
                               [ids, ids[:9], ids[:17], ids[:23]])
        r["diff_batch"] = float((both[0] - ref[2][0]).abs().max())
    print(f"  許容誤差 1e-4 でロジットが一致: "
          f"{sum(r['close'] for r in rows)}/{len(rows)}")
    print("  参考: ほかの 3 件とまとめて計算したときとの差の最大値: "
          + "  ".join(f"{r['diff_batch']:.1e}" for r in rows))

    x = np.array([len(r["prefix"]) for r in rows])
    saved = np.array([med(r["t"]["none"]) - med(r["t"]["mem"])
                      for r in rows])
    slope, icpt = np.polyfit(x, saved, 1)
    print("\n  省けた時間（なし - メモリ）: "
          + "  ".join(f"{s:.2f}" for s in saved) + " ミリ秒")
    print(f"  直線を当てはめると、前置き 1 トークンあたり {slope:.3f} ミリ秒"
          f"（切片 {icpt:+.2f} ミリ秒）")
    with open("ch19/out/prefix_cache.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_prefix"] + [f"{k}_ms" for k in WAYS]
                   + ["dec_none_ms", "dec_mem_ms"])
        for r in rows:
            w.writerow([len(r["prefix"])]
                       + [f"{med(r['t'][k]):.3f}" for k in WAYS]
                       + [f"{med(r['dec'][k]):.3f}" for k in WAYS[:2]])

    print("\n[2] どこまで使い回せるか（保存したのは 8 往復、112 トークン）")
    cache = torch.load(NEAR)
    base = history_ids(TURNS, NEW_QUESTION)
    edited = list(TURNS)
    edited[4] = (TURNS[4][0], "82です。")  # 5 往復目の応答を書き換える
    first = [tok.special_tokens["<|tool|>"]] + base[1:]
    cases = [("続きを足しただけ", base),
             ("5 往復目の応答を 1 文字変えた",
              history_ids(edited, NEW_QUESTION)),
             ("先頭のトークンを変えた", first),
             ("最初の 1 往復を削った", history_ids(TURNS[1:], NEW_QUESTION))]
    engine = Engine(model, n_slots=1)
    times = {name: [] for name, _ in cases}
    for _ in range(REPEAT):
        for name, ids in cases:
            t0 = time.perf_counter()
            pick(read_cache(engine, cache, ids))
            times[name].append((time.perf_counter() - t0) * 1000)
    print("  要求  使い回せる  計算する  1 個目まで（ミリ秒）")
    for name, ids in cases:
        n = common_prefix(cache["ids"], ids)
        print(f"  {len(ids):4d}    {n:4d}       {len(ids) - n:4d}"
              f"       {med(times[name]):5.2f}    {name}")

    print("\n[3] 8 回の質問を順に送る（毎回、履歴の全体を送り直す）")
    runs = {False: [], True: []}
    for _ in range(REPEAT):
        for c in runs:
            runs[c].append(chat(c))
    for c, name in [(False, "キャッシュなし"), (True, "前の回の KV を残す")]:
        print(f"  {name}: プリフィルしたトークン 計 {runs[c][-1][0]:3d}、"
              f"その時間 計 {med([r[1] for r in runs[c]]):5.1f} ミリ秒")
    print(f"  会話の全文が一致: {runs[False][-1][2] == runs[True][-1][2]}")
    os.remove(NEAR)
