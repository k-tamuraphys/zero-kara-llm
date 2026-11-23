"""コールドスタートの SFT：加減算の「電卓への式の書き方」だけを教える。
加減算の質問には、硬貨を投げて半分は電卓（式は a+b / a-b）、半分は暗算の
お手本を付ける。どちらを使うべきかの情報は入れない（それは RL が学ぶ）。
  python cold_start.py  → checkpoints/ch24_cold.pt, out/cold_start.txt
"""
import csv, random, re, time
import torch
from env import *
from llmbook.chat import collate, encode_chat_example
from llmbook.train import lr_at, make_optimizer

torch.set_num_threads(2)
torch.manual_seed(0)
t0 = time.time()
model = load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt")

# [snippet: cold-data]
def add_example(rng):
    m, a = make_chat_example(rng, ops="+-")
    q = m[0]["content"]
    if rng.random() < 0.5:                      # 硬貨を投げて半分だけ電卓
        x, y = re.findall(r"\d+", q)[:2]
        op = "-" if ("-" in q or "引" in q) else "+"
        m = [m[0], {"role": "tool", "content": f"{x}{op}{y}"},
             {"role": "result", "content": a},
             {"role": "assistant", "content": f"{a}です。"}]
    return m

rng = random.Random(7)
data = [add_example(rng) for _ in range(1500)] + \
       [make_tool_example(rng, k)[0]
        for k in ["mul", "two"] for _ in range(750)]
examples = [encode_chat_example(tok, m, learn_role=True) for m in data]
# [end: cold-data]

steps, bs, max_lr = 200, 32, 1e-4
opt = make_optimizer(model, max_lr, weight_decay=0.0)
for step in range(steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step, max_lr, warmup_steps=20, max_steps=steps)
    x, y = collate(rng.sample(examples, bs), PAD)
    model.train()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0 or step == steps - 1:
        print(f"step {step}: 損失 {loss.item():.3f}", flush=True)
save_checkpoint(model, CKPT_DIR / "ch24_cold.pt", steps=steps, lr=max_lr)

res, trs = evaluate(model, eval_sets())
print("コールドスタート後（貪欲）")
for k, v in res.items():
    print(f"  {k}: 正答率 {v['acc']:.3f}  電卓使用率 {v['tool']:.3f}"
          f"  平均呼び出し {v['calls']:.2f}")
with open(OUT_DIR / "eval_cold.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "question", "answer", "calls", "correct", "traj"])
    for k, ts in trs.items():
        for t in ts:
            w.writerow([k, t.task.question, t.task.answer, t.n_calls,
                        int(reward(t, 0) == 1.0), "".join(t.texts)])
print(f"[{time.time() - t0:.0f} 秒]")
