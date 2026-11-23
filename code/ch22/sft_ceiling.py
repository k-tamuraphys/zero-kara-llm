"""診断：sft.pt から、正解付きの新しい問題で SFT をさらに続けたら
テスト 1,000 問（種 777、貪欲）の正答率はどこまで上がるか。
強化学習の結果を読むための「このモデルの大きさでの到達の目安」。
  python ch22/sft_ceiling.py --threads 1      # 約 2 分
"""
import argparse
import random
import time

import torch

from common import OUT_DIR, PAD, load_sft, tok
from llmbook.chat import collate, encode_chat_example, make_chat_example
from rl_lib import EVAL_TEST, eval_acc

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--B", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
rng = random.Random(2024)  # 訓練（種 0）とも評価（種 777, 555）とも別

model = load_sft()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
t0, lines = time.time(), []
for step in range(args.steps + 1):
    if step % 100 == 0:
        acc = eval_acc(model, EVAL_TEST)
        lines.append(f"step {step:4d}  test1000 {acc:.3f}  "
                     f"({time.time()-t0:.0f}s)")
        print(lines[-1], flush=True)
    if step == args.steps:
        break
    batch = [encode_chat_example(tok, make_chat_example(rng)[0])
             for _ in range(args.B)]          # 毎回新しい問題（使い回さない）
    x, y = collate(batch, PAD)
    model.train()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
with open(OUT_DIR / f"sft_ceiling{args.tag}.txt", "w") as f:
    print(f"SFT 継続: lr {args.lr}, バッチ {args.B}, {args.steps} ステップ "
          f"（新しい問題 {args.B * args.steps:,} 件）", file=f)
    print("\n".join(lines), file=f)
