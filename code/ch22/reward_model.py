"""報酬モデル：選好の対から「応答の良さ」を 1 つの数で返すモデルを学ぶ
（Bradley–Terry モデル）。学んだ報酬が「何を」測っているかも調べる。"""
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_sft, tok, prompt_ids, device
from prefs import make_prefs, pad_batch

torch.manual_seed(0)


# [snippet: reward-model]
class RewardModel(nn.Module):
    """SFT 済みの GPT の胴体に、スカラーを出す頭を付けたもの。"""
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.config.n_embd, 1)

    def forward(self, seqs: list[list[int]]) -> torch.Tensor:
        x, lens = pad_batch(seqs)                       # (B, T)
        g = self.gpt
        h = g.tok_emb(x) + g.pos_emb(torch.arange(x.size(1), device=device))
        for block in g.blocks:
            h = block(h)
        h = g.ln_f(h)                                   # (B, T, d)
        last = h[torch.arange(len(seqs)), torch.tensor(lens) - 1]  # (B, d)
        return self.head(last).squeeze(1)               # (B,) 報酬 r_φ
# [end: reward-model]


rm = RewardModel(load_sft()).to(device)
opt = torch.optim.Adam(rm.parameters(), lr=3e-5)
train, test = make_prefs(2000, seed=1), make_prefs(200, seed=2)


# [snippet: bt-loss]
def bt_loss(batch):
    """−log σ(r_φ(x, y_w) − r_φ(x, y_l)) の平均と、正しく並べた割合。"""
    r_w = rm([p + w for p, w, _ in batch])
    r_l = rm([p + l for p, _, l in batch])
    return -F.logsigmoid(r_w - r_l).mean(), (r_w > r_l).float().mean()
# [end: bt-loss]


t0 = time.time()
for step in range(101):
    if step % 25 == 0:
        with torch.no_grad():
            loss, acc = bt_loss(test)
        print(f"step {step:3d}  test loss {loss:.3f}  test acc {acc:.3f}")
    if step == 100:
        break
    loss, _ = bt_loss(random.Random(step).sample(train, 16))
    opt.zero_grad()
    loss.backward()
    opt.step()

# 学んだ報酬は「正しさ」を測っているか？ 12+35 の正解は 47。
p = prompt_ids("12+35は？")
probes = ["47です。", "57です。", "答えは47です。", "答えは57です。",
          "答えは答えは47です。"]
with torch.no_grad():
    scores = rm([p + tok.encode(s + "<|end|>") for s in probes])
for s, v in zip(probes, scores.tolist()):
    print(f"  r({s}) = {v:+.2f}")
print(f"total {time.time()-t0:.0f}s")
