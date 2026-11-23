"""汚染の実験: 学習で見た問題と、見ていない問題で、正答率はどれだけ違うか。"""

import random
import re

import torch

from common import CKPT_DIR, answer_all, load, make_problems
from llmbook.arith import drill_text
from llmbook.chat import collate, encode_chat_example
from llmbook.train import lr_at, make_optimizer, save_checkpoint


def key(p: dict) -> tuple:
    """言い回しを無視した「同じ計算」の目印。"""
    return (p["a"], p["op"], p["b"])


def accuracy(model, tok, problems: list[dict]) -> float:
    replies = answer_all(model, tok, [p["question"] for p in problems])
    n_ok = sum(r == p["target"] for r, p in zip(replies, problems))
    return n_ok / len(problems)


model, tok = load("sft")

# ---- (1) 試験問題は、学習データとどれだけ重なっているか ----
sft_train = make_problems(3000, seed=0)  # sft.pt の学習データと同じ 3,000 件
test = make_problems(1000, seed=777)     # 試験問題（種が違う）
train_questions = {p["question"] for p in sft_train}
train_keys = {key(p) for p in sft_train}
drill_keys = set()                       # 事前学習の計算ドリル 12 万行
for line in drill_text(120_000, seed=0).splitlines():
    a, op, b = re.match(r"(\d+)([+-])(\d+)=", line).groups()
    drill_keys.add((int(a), op, int(b)))

# [snippet: overlap]
same_text = [p for p in test if p["question"] in train_questions]
same_calc = [p for p in test if key(p) in train_keys]
in_drill = [p for p in test if key(p) in drill_keys]
# [end: overlap]
print("試験問題 1,000 問（種 777）のうち")
print(f"  SFT の学習データに同じ質問文がある     : {len(same_text):4d} 問")
print(f"  SFT の学習データに同じ計算がある       : {len(same_calc):4d} 問")
print(f"  事前学習のドリルに同じ計算がある       : {len(in_drill):4d} 問")
for d in (1, 2, 3):
    sub = [p for p in test if p["digits"] == d]
    hit = [p for p in sub if key(p) in drill_keys or key(p) in train_keys]
    print(f"    {d} 桁の問題 {len(sub):3d} 問のうち、どこかで見た計算: "
          f"{len(hit):3d} 問（{len(hit) / len(sub):.0%}）")


# ---- (2) 見た問題と、条件をそろえた見ていない問題 ----
def twin(p: dict, rng: random.Random, seen: set) -> dict:
    """p と桁数・演算・言い回しが同じで、数だけが違う未見の問題を作る。"""
    da, db = len(str(p["a"])), len(str(p["b"]))
    while True:
        a = rng.randint(10 ** (da - 1) if da > 1 else 0, 10 ** da - 1)
        b = rng.randint(10 ** (db - 1) if db > 1 else 0, 10 ** db - 1)
        if (p["op"] == "+" or a >= b) and (a, p["op"], b) not in seen:
            break
    question = re.sub(r"\d+", "{}", p["question"]).format(a, b)
    answer = a + b if p["op"] == "+" else a - b
    return {**p, "a": a, "b": b, "question": question,
            "target": f"{answer}です。"}


rng = random.Random(1)
seen_all = train_keys | drill_keys
print("\n見た問題と、条件をそろえた見ていない問題の正答率"
      "（3 桁の問題 1,000 問ずつ）")
# (2a) SFT の学習データにあった質問
seen = list({p["question"]: p for p in sft_train
             if p["digits"] == 3}.values())[:1000]
unseen = [twin(p, rng, seen_all) for p in seen]
acc_s, acc_u = accuracy(model, tok, seen), accuracy(model, tok, unseen)
print(f"  SFT で見た質問      : {acc_s:.1%}"
      f"   数だけ変えた未見の問題: {acc_u:.1%}")
# (2b) 事前学習のドリルにだけ出てきた計算
pool = [p for p in make_problems(20000, seed=5)
        if p["digits"] == 3 and key(p) in drill_keys
        and key(p) not in train_keys]
seen = list({key(p): p for p in pool}.values())[:1000]
unseen = [twin(p, rng, seen_all) for p in seen]
acc_s, acc_u = accuracy(model, tok, seen), accuracy(model, tok, unseen)
print(f"  ドリルで見た計算    : {acc_s:.1%}"
      f"   数だけ変えた未見の問題: {acc_u:.1%}")

# ---- (3) 試験問題そのものを学習データに混ぜてしまったら ----
torch.manual_seed(0)
rng = random.Random(0)
leaked = [encode_chat_example(tok, [
    {"role": "user", "content": p["question"]},
    {"role": "assistant", "content": p["target"]}]) for p in test]
fresh = make_problems(1000, seed=778)  # 学習にも試験にも使っていない問題
print("\n試験問題 1,000 問を学習データに混ぜて追加学習する")
print(f"  学習前  : 試験問題 {accuracy(model, tok, test):.1%}"
      f"   新しい問題 {accuracy(model, tok, fresh):.1%}")
max_steps, pad_id = 300, tok.special_tokens["<|pad|>"]
model.train()
optimizer = make_optimizer(model, 3e-4, weight_decay=0.0)
for step in range(max_steps):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step, 3e-4, warmup_steps=30, max_steps=max_steps)
    x, y = collate(rng.sample(leaked, 32), pad_id)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
model.eval()
print(f"  学習後  : 試験問題 {accuracy(model, tok, test):.1%}"
      f"   新しい問題 {accuracy(model, tok, fresh):.1%}")
save_checkpoint(model, CKPT_DIR / "ch19_leak.pt")
