"""第 5 章: 勾配クリッピング。

(1) 自作の関数を手計算の例と torch の関数で確かめる。
(2) 隠れ層 10 枚（He の初期化）を学習率 0.5 の SGD で学習させ、
    クリッピングの有無を比べる。
    cd code && python ch05/clip.py > ch05/out/clip.txt
"""
import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, evaluate, load_spiral

torch.set_num_threads(1)
X, Y, Xt, Yt = load_spiral()


# [snippet: clip]
def clip_grad_norm(params, max_norm):
    """全パラメータの勾配をまとめたノルムが max_norm を超えたら縮める。"""
    params = list(params)
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    if total > max_norm:
        for p in params:
            p.grad *= max_norm / total       # 向きは変えず、長さだけ縮める
    return total.item()                      # 縮める前のノルムを返す
# [end: clip]


w = torch.zeros(2, requires_grad=True)
w.grad = torch.tensor([3.0, 4.0])
norm = clip_grad_norm([w], max_norm=1.0)
print(f"縮める前のノルム {norm:.1f} → 勾配"
      f" {[round(v, 2) for v in w.grad.tolist()]}")
w.grad = torch.tensor([0.3, 0.4])
norm = clip_grad_norm([w], max_norm=1.0)
print(f"縮める前のノルム {norm:.1f} → 勾配"
      f" {[round(v, 2) for v in w.grad.tolist()]}（そのまま）")
w.grad = torch.tensor([3.0, 4.0])
torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
print(f"torch.nn.utils.clip_grad_norm_ → 勾配"
      f" {[round(v, 4) for v in w.grad.tolist()]}")
print("---")

STEPS, LR = 500, 0.5


def train(max_norm):
    torch.manual_seed(0)
    model = DeepMLP(10, init="he")
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    losses, norms = [], []
    for step in range(STEPS):
        loss = F.cross_entropy(model(X), Y)
        optimizer.zero_grad()
        loss.backward()
        # [snippet: clip-use]
        norm = clip_grad_norm(model.parameters(), max_norm)  # backward の後
        optimizer.step()                                     # step の前
        # [end: clip-use]
        losses.append(loss.item())
        norms.append(norm)
    return losses, norms, evaluate(model, X, Y), evaluate(model, Xt, Yt)


results = {"なし": train(float("inf")), "あり（上限 1.0）": train(1.0)}
print(f"隠れ層 10 枚、He の初期化、SGD（学習率 {LR}）、{STEPS} ステップ")
for name, (losses, norms, tr, te) in results.items():
    big = sum(n > 1.0 for n in norms)
    print(f"クリッピング{name}")
    print(f"  勾配のノルムの最大 {max(norms):.1f}（1.0 を超えた回数 {big}）")
    print(f"  途中の損失の最大 {max(losses):.2f}")
    print(f"  最後の訓練損失 {tr[0]:.4f}（正解率 {tr[1] * 100:.0f}%）"
          f"  テスト正解率 {te[1] * 100:.0f}%")
with open("ch05/out/clip.csv", "w") as f:
    f.write("step,loss_off,norm_off,loss_on,norm_on\n")
    off, on = results["なし"], results["あり（上限 1.0）"]
    for i in range(STEPS):
        f.write(f"{i},{off[0][i]:.5f},{off[1][i]:.5f},"
                f"{on[0][i]:.5f},{on[1][i]:.5f}\n")
