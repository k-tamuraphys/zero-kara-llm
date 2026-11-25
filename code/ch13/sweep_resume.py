"""resume_demo.py と同じ 3 通りの学習で、40 ステップすべての損失を記録する。

実行: python ch13/sweep_resume.py  （約 20 秒）
出力: ch13/out/sweep_resume.csv（列は step, straight, resumed, fresh）
"""

import copy

import torch

from ch13.common import load_bin
from llmbook.model import GPT, GPTConfig
from llmbook.train import get_batch, make_optimizer

torch.set_num_threads(2)
train_data = load_bin("train")
config = GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=32)


def run(model, optimizer, start, stop):
    """start から stop の手前まで学習し、各ステップの損失を返す。"""
    losses = []
    for step in range(start, stop):
        x, y = get_batch(train_data, 8, config.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# (a) 止めずに 40 ステップ
torch.manual_seed(0)
model = GPT(config)
straight = run(model, make_optimizer(model, 1e-3), 0, 40)

# (b) 20 ステップで状態を写し取り、新しいモデルとオプティマイザで続ける
torch.manual_seed(0)
model = GPT(config)
opt = make_optimizer(model, 1e-3)
first = run(model, opt, 0, 20)
weights = copy.deepcopy(model.state_dict())
opt_state = copy.deepcopy(opt.state_dict())
rng = torch.get_rng_state()
model2 = GPT(config)
model2.load_state_dict(weights)
opt2 = make_optimizer(model2, 1e-3)
opt2.load_state_dict(opt_state)
torch.set_rng_state(rng)
resumed = first + run(model2, opt2, 20, 40)

# (c) 重みと乱数だけ戻し、オプティマイザは作り直す
model3 = GPT(config)
model3.load_state_dict(weights)
torch.set_rng_state(rng)
fresh = first + run(model3, make_optimizer(model3, 1e-3), 20, 40)

with open("ch13/out/sweep_resume.csv", "w") as f:
    f.write("step,straight,resumed,fresh\n")
    for s in range(40):
        f.write(f"{s},{straight[s]:.4f},{resumed[s]:.4f},{fresh[s]:.4f}\n")
for s in [20, 21, 30, 39]:  # resume_demo.txt と同じ値になるかの確認
    print(f"step {s}: {straight[s]:.4f} / {resumed[s]:.4f} / {fresh[s]:.4f}")
