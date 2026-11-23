"""文字単位のミニGPT を漱石コーパスで学習する（CPU で数分）。

実行: python ch10/train_char_gpt.py
出力: ch10/out/train_char_gpt.txt（記録）, train_char_gpt.csv（学習曲線）,
      samples.txt（学習の途中の生成例）, ../checkpoints/ch10_char_gpt.pt
"""

import csv
import math
import time
from collections import Counter
from pathlib import Path

import torch

from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.model import GPT, GPTConfig
from llmbook.tokenizer import CharTokenizer
from llmbook.train import estimate_loss, get_batch, lr_at, make_optimizer

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)
torch.manual_seed(0)

# [snippet: data]
text = load_corpus()
train_text, val_text = train_val_split(text)  # 末尾 5% を検証用に
tok = CharTokenizer(text)  # 1 文字 = 1 トークン
train_data = torch.tensor(tok.encode(train_text))  # (約 229 万,)
val_data = torch.tensor(tok.encode(val_text))
print(f"語彙 {tok.vocab_size} 字, 訓練 {len(train_data):,} 字, "
      f"検証 {len(val_data):,} 字")
# [end: data]


# [snippet: baselines]
def bigram_val_loss(train_ids, val_ids, V, alpha=0.1):
    """数え上げのバイグラム（第 6 章）の検証損失。alpha は加算平滑化。"""
    pair = Counter(zip(train_ids, train_ids[1:]))
    first = Counter(train_ids[:-1])
    nll = 0.0
    for a, b in zip(val_ids, val_ids[1:]):
        p = (pair[(a, b)] + alpha) / (first[a] + alpha * V)
        nll -= math.log(p)
    return nll / (len(val_ids) - 1)
# [end: baselines]


tr, va = train_data.tolist(), val_data.tolist()
V = tok.vocab_size
uni = Counter(tr)
uni_loss = -sum(math.log((uni[b] + 0.1) / (len(tr) + 0.1 * V))
                for b in va) / len(va)
big_loss = bigram_val_loss(tr, va, V)
print(f"一様分布 ln V      : {math.log(V):.3f}")
print(f"ユニグラム（頻度） : {uni_loss:.3f}")
print(f"バイグラム         : {big_loss:.3f}")

# [snippet: model]
config = GPTConfig(vocab_size=tok.vocab_size, block_size=64,
                   n_layer=4, n_head=4, n_embd=128)
model = GPT(config)
print(f"パラメータ数 {model.num_params():,}")
# [end: model]

# [snippet: train-loop]
max_steps, batch_size, max_lr = 1500, 32, 2e-3
optimizer = make_optimizer(model, max_lr)
prompt = torch.tensor([tok.encode("　吾輩は")])
log, samples, t0 = [], [], time.time()
for step in range(max_steps + 1):
    if step % 100 == 0:  # 定期的に損失を測り、生成例を残す
        tl = estimate_loss(model, train_data, batch_size)
        vl = estimate_loss(model, val_data, batch_size)
        log.append({"step": step, "train_loss": round(tl, 4),
                    "val_loss": round(vl, 4)})
        print(f"step {step:4d} | train {tl:.3f} | val {vl:.3f} | "
              f"{time.time() - t0:4.0f} 秒")
        if step in (0, 100, 300, 1500):
            torch.manual_seed(1)
            out = model.generate(prompt, 40, temperature=0.8, top_k=50)
            s = tok.decode(out[0].tolist()).replace("\n", "／")  # 改行は／
            samples.append(f"step {step:4d}: {s}")
    if step == max_steps:
        break
    for group in optimizer.param_groups:  # ウォームアップ + コサイン
        group["lr"] = lr_at(step, max_lr, 100, max_steps)
    x, y = get_batch(train_data, batch_size, config.block_size)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
# [end: train-loop]

print(f"所要時間 {time.time() - t0:.0f} 秒")
with open(OUT / "train_char_gpt.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0]))
    w.writeheader()
    w.writerows(log)
(OUT / "samples.txt").write_text("\n".join(samples) + "\n", encoding="utf-8")
torch.save({"config": vars(config), "state_dict": model.state_dict()},
           DATA_DIR.parent / "checkpoints" / "ch10_char_gpt.pt")
