"""PyTorch の nn.Module と optim で MLP を作り、渦巻き 300 点を分類する。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_spiral

torch.manual_seed(0)
torch.set_num_threads(2)


# [snippet: model]
class MLP(nn.Module):
    """隠れ層 2 枚の多層パーセプトロン。"""

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)  # 重みとバイアスを持つ
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

    def forward(self, x):  # x: (N, n_in)
        h = torch.relu(self.fc1(x))  # (N, n_hidden)
        h = torch.relu(self.fc2(h))  # (N, n_hidden)
        return self.fc3(h)  # (N, n_out) ロジット
# [end: model]


# [snippet: setup]
X, Y = make_spiral(n_per_class=100, seed=0)  # 訓練データ 300 点
X, Y = torch.tensor(X), torch.tensor(Y)  # (300, 2), (300,)
X_test, Y_test = make_spiral(n_per_class=100, seed=1)  # テスト用（別の乱数）
X_test, Y_test = torch.tensor(X_test), torch.tensor(Y_test)

model = MLP(2, 32, 3)
for name, p in model.named_parameters():
    print(f"{name:11s} 形 {tuple(p.shape)}")
print("パラメータ数:", sum(p.numel() for p in model.parameters()))
# [end: setup]
# [snippet: optimizer]
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
# [end: optimizer]


# [snippet: evaluate]
def evaluate(model, X, Y):
    """損失と正解率を測る（学習はしない）。"""
    with torch.no_grad():  # この中では計算グラフを記録しない
        logits = model(X)
        loss = F.cross_entropy(logits, Y).item()
        acc = (logits.argmax(dim=1) == Y).float().mean().item()
    return loss, acc
# [end: evaluate]


log = open("ch04/out/mlp_torch_log.csv", "w")
log.write("step,train_loss,test_loss,train_acc,test_acc\n")
for step in range(1001):
    if step % 10 == 0:  # 記録（学習そのものには関係しない）
        (tr_loss, tr_acc) = evaluate(model, X, Y)
        (te_loss, te_acc) = evaluate(model, X_test, Y_test)
        log.write(f"{step},{tr_loss:.5f},{te_loss:.5f},"
                  f"{tr_acc:.4f},{te_acc:.4f}\n")
    if step % 500 == 0:
        print(f"step {step:4d}  訓練: 損失 {tr_loss:.4f} 正解率 {tr_acc:.3f}"
              f"   テスト: 損失 {te_loss:.4f} 正解率 {te_acc:.3f}")
    # [snippet: loop]
    logits = model(X)  # (1) 順伝播
    loss = F.cross_entropy(logits, Y)  # (2) 損失
    optimizer.zero_grad()  # (3) 勾配を 0 に戻す
    loss.backward()  # (4) 逆伝播
    optimizer.step()  # (5) パラメータを更新
    # [end: loop]
log.close()

# 図用: 格子点ごとの予測クラス（決定領域）
n = 60
g = torch.linspace(-1.15, 1.15, n)
grid = torch.stack([g.repeat(n), g.repeat_interleave(n)], dim=1)  # (n*n, 2)
with torch.no_grad():
    cls = model(grid).argmax(dim=1)
with open("ch04/out/mlp_regions.csv", "w") as f:
    f.write("x1,x2,c\n")
    for (x1, x2), c in zip(grid.tolist(), cls.tolist()):
        f.write(f"{x1:.4f},{x2:.4f},{c}\n")
