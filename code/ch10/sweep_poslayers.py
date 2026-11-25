"""位置埋め込みを消したとき、語順の入れ替えがロジットに届くかを、
ブロックの数を変えて調べる（コード 10.1 と同じ比較。学習前のモデル）。

実行: python ch10/sweep_poslayers.py   （数秒）
出力: ch10/out/sweep_poslayers.txt, sweep_poslayers.csv
"""

from pathlib import Path

import torch

from llmbook.model import GPT, GPTConfig

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)
LAYERS = [1, 2, 3, 4]  # ブロックの数 L
SEEDS = [0, 1, 2, 3, 4]  # 初期値の乱数の種（ばらつきを見るため 5 通り）
a = torch.tensor([[5, 7, 9]])  # 「5 7 9」と
b = torch.tensor([[7, 5, 9]])  # 先頭 2 つを入れ替えた「7 5 9」


def diff(model: GPT) -> float:
    """2 つの入力の、最後の位置のロジットの差の絶対値の最大。"""
    with torch.no_grad():
        za, zb = model(a)[0][0, -1], model(b)[0][0, -1]
    return float((za - zb).abs().max())


rows, lines = [], []
for L in LAYERS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = GPT(GPTConfig(vocab_size=100, block_size=8, n_layer=L,
                              n_head=2, n_embd=16)).eval()
        with_pos = diff(model)
        with torch.no_grad():
            model.pos_emb.weight.zero_()  # 位置埋め込みを 0 にして消す
        without = diff(model)
        rows.append((L, seed, with_pos, without))
        lines.append(f"L={L} 種={seed}: 位置埋め込みあり {with_pos:.2e}, "
                     f"なし {without:.2e}")

with open(OUT / "sweep_poslayers.csv", "w", encoding="utf-8") as f:
    # 図で 2 つの系列が重ならないよう、横の位置を少しずらした列も書く
    f.write("n_layer,seed,with_pos,without_pos,x_with,x_without\n")
    for L, seed, w, wo in rows:
        f.write(f"{L},{seed},{w:.3e},{wo:.3e},{L - 0.12:.2f},"
                f"{L + 0.12:.2f}\n")
# 同じトークンだけの列「5 5 5 5」では、位置埋め込みなしだと位置を区別
# できるか（4 ブロック、種 0）。各位置のロジットと位置 1 との差の最大。
torch.manual_seed(0)
model = GPT(GPTConfig(vocab_size=100, block_size=8, n_layer=4,
                      n_head=2, n_embd=16)).eval()
with torch.no_grad():
    model.pos_emb.weight.zero_()
    z = model(torch.tensor([[5, 5, 5, 5]]))[0][0]  # (4, V)
    gap = (z - z[0]).abs().max(dim=-1).values
lines.append("「5 5 5 5」、L=4、位置埋め込みなし: 位置 1 との差の最大 "
             + ", ".join(f"{float(g):.1e}" for g in gap))
(OUT / "sweep_poslayers.txt").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
print("\n".join(lines))
