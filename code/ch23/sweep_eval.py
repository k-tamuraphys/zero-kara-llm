"""ch23_tool_sft.pt のエージェントの正答率が、条件でどれだけ動くかを測る。

(1) 評価の問題の種を 5 通りに変える（各 50 問）。50 問で測った正答率の
    ぶれ（本文 23.5 節の「約 5 ポイント」）を実際に見るため。
(2) 生成の温度を 5 通りに変える（各 100 問、問題は本文と同じ種 12345）。
    温度を上げると、電卓の呼び方と正答率がどう変わるかを見るため。
モデル・ループ・電卓は本文のまま。学習はしない。
実行:  python ch23/sweep_eval.py > ch23/out/sweep_eval.txt   （約 5 分）
出力:  ch23/out/sweep_eval_seeds.csv, ch23/out/sweep_eval_temp.csv
"""

import csv

import torch

from common import CKPT_DIR, OUT_DIR, device
from mini_agent import MiniGPTLLM, agent_accuracy
from llmbook.train import load_checkpoint

kinds = ["add", "mul", "two"]
model = load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt", device)

# (1) 問題の種だけを変える（温度 0 なので、モデルの出力に乱数は入らない）
seeds = [12345, 12346, 12347, 12348, 12349]
llm = MiniGPTLLM(model, temperature=0.0)
rows = []
for seed in seeds:
    r = {k: agent_accuracy(llm, k, n_problems=50, seed=seed)["acc"]
         for k in kinds}
    rows.append({"seed": seed} | r)
    print("種", seed, " ".join(f"{k} {r[k]:5.1%}" for k in kinds),
          flush=True)
with open(OUT_DIR / "sweep_eval_seeds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed"] + kinds)
    w.writeheader()
    w.writerows(rows)

# (2) 温度だけを変える（サンプリングの乱数の種は温度ごとに 0 に戻す）
temps = [0.0, 0.3, 0.6, 1.0, 1.5]
rows = []
for T in temps:
    torch.manual_seed(0)
    llm = MiniGPTLLM(model, temperature=T)
    row = {"temp": T}
    for k in kinds:
        r = agent_accuracy(llm, k, n_problems=100)
        row[f"{k}_acc"], row[f"{k}_tool"] = r["acc"], r["tool_rate"]
    rows.append(row)
    print(f"温度 {T:.1f} " + " ".join(
        f"{k} 正答率 {row[k + '_acc']:5.1%} 電卓 {row[k + '_tool']:5.1%}"
        for k in kinds), flush=True)
with open(OUT_DIR / "sweep_eval_temp.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
