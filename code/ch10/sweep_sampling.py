"""生成の設定（温度・top-k・top-p）を変えると、書く文章の性質が
どう変わるかを、基盤モデル base.pt で数える小さなスイープ。

各設定で「　吾輩は」の続きを 30 トークンずつ 20 本生成し、次を平均する。
  bad  : 文字にならないバイト列（□）の個数（1 本あたり）
  rep  : 1 本の中で、前に出たのと同じ 3 トークンの並びが出た割合
  nll  : 選んだトークンの、温度 1 のモデルでの -log 確率の平均
実行: python ch10/sweep_sampling.py   （2 スレッドで 1 分ほど）
出力: ch10/out/sweep_sampling.txt, sweep_sampling.csv, sweep_sampling_wide.csv
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()
TAUS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]  # 温度（0 は貪欲法）
FILTERS = [("none", {}), ("topk10", dict(top_k=10)),
           ("topp09", dict(top_p=0.9))]
N, NEW = 20, 30  # 1 設定あたりの本数と、1 本のトークン数
start = torch.tensor([tok.encode("\n　吾輩は")] * N)  # (N, 2)


def measure(out: torch.Tensor) -> tuple[float, float, float]:
    """生成した (N, 2+NEW) の列から 3 つの指標を計算する。"""
    with torch.no_grad():
        logits = model(out)[0][:, :-1]  # 位置 t のロジットで t+1 を測る
    logp = F.log_softmax(logits, dim=-1)
    tgt = out[:, 1:]
    nll = -logp.gather(2, tgt.unsqueeze(2)).squeeze(2)[:, -NEW:]
    bad, rep = 0.0, 0.0
    for row in out[:, start.size(1):].tolist():
        bad += tok.decode(row).count("�")  # 置換文字 = □
        tri = [tuple(row[i:i + 3]) for i in range(len(row) - 2)]
        seen, dup = set(), 0
        for g in tri:
            dup += g in seen
            seen.add(g)
        rep += dup / len(tri)
    return bad / N, rep / N, float(nll.mean())


rows, lines = [], []
for name, kw in FILTERS:
    for tau in TAUS:
        torch.manual_seed(0)  # どの設定でも同じ乱数の列から始める
        out = model.generate(start, max_new_tokens=NEW, temperature=tau,
                             **kw)
        bad, rep, nll = measure(out)
        rows.append((name, tau, bad, rep, nll))
        lines.append(f"{name:7s} 温度 {tau:.1f}: □ {bad:5.2f} 個, "
                     f"繰り返し {rep:.3f}, -log 確率 {nll:.3f}")

with open(OUT / "sweep_sampling.csv", "w", encoding="utf-8") as f:
    f.write("filter,tau,bad,rep,nll\n")
    for name, tau, bad, rep, nll in rows:
        f.write(f"{name},{tau},{bad:.3f},{rep:.4f},{nll:.4f}\n")
# 図にしやすいよう、温度ごとに 1 行の横長の表も書く
with open(OUT / "sweep_sampling_wide.csv", "w", encoding="utf-8") as f:
    names = [n for n, _ in FILTERS]
    f.write("tau," + ",".join(f"{m}_{n}" for m in ["bad", "rep", "nll"]
                               for n in names) + "\n")
    table = {(n, t): (b, r, l) for n, t, b, r, l in rows}
    for tau in TAUS:
        vals = [table[(n, tau)][k] for k in range(3) for n in names]
        f.write(f"{tau}," + ",".join(f"{v:.4f}" for v in vals) + "\n")
(OUT / "sweep_sampling.txt").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
print("\n".join(lines))
