"""チェックポイントから再開した学習が、止めなかった学習と同じになるか確かめる。

実行: python ch13/resume_demo.py  （約 20 秒）
出力: ch13/out/resume_demo.txt
"""

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
opt = make_optimizer(model, 1e-3)
straight = run(model, opt, 0, 40)

# [snippet: save-state]
# (b) 20 ステップで止めて、学習の「状態」を丸ごと保存する
torch.manual_seed(0)
model = GPT(config)
opt = make_optimizer(model, 1e-3)
first = run(model, opt, 0, 20)
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
            "step": 20, "rng": torch.get_rng_state()},
           "ch13/out/resume_tmp.pt")
# [end: save-state]

# [snippet: load-state]
# 別のプロセスで再開したつもりで、新しく作ったモデルに読み込む
ckpt = torch.load("ch13/out/resume_tmp.pt", weights_only=True)
model2 = GPT(config)
model2.load_state_dict(ckpt["model"])
opt2 = make_optimizer(model2, 1e-3)
opt2.load_state_dict(ckpt["optimizer"])  # Adam の m と v も戻す
torch.set_rng_state(ckpt["rng"])  # データの選び方も続きから
resumed = first + run(model2, opt2, ckpt["step"], 40)
# [end: load-state]

# (c) 比較: オプティマイザの状態を戻さずに再開した場合
model3 = GPT(config)
model3.load_state_dict(ckpt["model"])
torch.set_rng_state(ckpt["rng"])
fresh = first + run(model3, make_optimizer(model3, 1e-3), 20, 40)

for s in [20, 21, 30, 39]:
    print(f"step {s}: 止めない {straight[s]:.4f} / 完全に再開 "
          f"{resumed[s]:.4f} / 重みだけ再開 {fresh[s]:.4f}")
diff = max(abs(a - b) for a, b in zip(straight, resumed))
print(f"止めない場合と完全に再開した場合の差の最大: {diff:.2e}")
