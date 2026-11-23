"""LoRA のマージ（推論コストが増えないこと）と、アダプタの付け替え。

実行:  python ch17/merge_and_swap.py   （先に train_lora.py 8 all を実行）
出力:  ch17/out/merge_and_swap.txt, checkpoints/ch17_lora_nyan.pt
"""

import statistics
import time

import torch

from common import CKPT_DIR, load_base, sft_examples, tok, train
from llmbook.chat import chat_reply, make_chat_example
from lora import add_lora, count_params, merge_all, unmerge_all

torch.manual_seed(0)
x = torch.randint(0, 4088, (8, 128))  # 速度と一致の確認に使う入力


def bench(model, n=15) -> float:
    """順伝播 1 回の時間（ミリ秒）の中央値。混雑のぶれを抑える。"""
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# [snippet: merge-check]
base = load_base().eval()
model = add_lora(load_base(), r=8, alpha=16).eval()
model.load_state_dict(torch.load(CKPT_DIR / "ch17_lora_r8_all.pt"),
                      strict=False)  # A と B だけを読み込む
with torch.no_grad():
    before = model(x)[0]
    merge_all(model)                 # W <- W + (alpha/r) B A
    after = model(x)[0]
print(f"マージ前後のロジットの最大差: {(before - after).abs().max():.2e}")
# [end: merge-check]
t_base, t_merged = bench(base), bench(model)
unmerge_all(model)
t_lora = bench(model)
print(f"順伝播の時間（8×128 トークン、中央値）: 基盤 {t_base:.1f} ms / "
      f"LoRA 未マージ {t_lora:.1f} ms / マージ後 {t_merged:.1f} ms")

# [snippet: swap]
nyan_model = add_lora(load_base(), r=8, alpha=16)
train(nyan_model, sft_examples(suffix="だニャ。"), max_lr=1e-3)
nyan = {k: v for k, v in nyan_model.state_dict().items()
        if k.endswith((".A", ".B"))}
torch.save(nyan, CKPT_DIR / "ch17_lora_nyan.pt")
desu = torch.load(CKPT_DIR / "ch17_lora_r8_all.pt")
for name, adapter in [("です調", desu), ("ニャ調", nyan)]:
    model.load_state_dict(adapter, strict=False)  # A, B だけ差し替える
    for q in ["12+35は？", "300から123を引くといくつですか？"]:
        print(f"[{name}] {q} -> {chat_reply(model, tok, q)}")
# [end: swap]
n_ok = 0
rng = __import__("random").Random(777)
for _ in range(500):
    messages, answer = make_chat_example(rng)
    n_ok += chat_reply(model, tok, messages[0]["content"]) == f"{answer}だニャ。"
print(f"ニャ調アダプタの正答率（500 問、語尾込みの完全一致）: {n_ok / 500:.1%}")
print(f"基盤モデル本体の大きさ: {count_params(load_base())[1]:,} パラメータ、"
      f"アダプタ 1 個: {sum(v.numel() for v in nyan.values()):,}")
