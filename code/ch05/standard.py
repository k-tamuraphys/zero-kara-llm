"""第 5 章: 標準セットで、隠れ層 50 枚の MLP を学習させる。

He の初期化＋レイヤー正規化＋残差接続のモデルを、
AdamW＋ウォームアップ／コサイン減衰＋勾配クリッピングで学習する。
最後に、同じモデルをただの SGD で学習させた結果と比べる。
    cd code && python ch05/standard.py > ch05/out/standard.txt
"""
import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, evaluate, load_spiral
from llmbook.train import lr_at, make_optimizer

torch.set_num_threads(1)
torch.manual_seed(0)
X, Y, Xt, Yt = load_spiral()

MAX_STEPS, WARMUP, MAX_LR, BATCH = 300, 30, 0.01, 32

# [snippet: standard]
model = DeepMLP(50, init="he", norm=True, residual=True)
optimizer = make_optimizer(model, MAX_LR, weight_decay=0.1)   # AdamW

for step in range(MAX_STEPS):
    lr = lr_at(step, MAX_LR, WARMUP, MAX_STEPS)   # 学習率スケジュール
    for group in optimizer.param_groups:
        group["lr"] = lr
    idx = torch.randint(0, len(X), (BATCH,))      # ミニバッチ
    logits = model(X[idx])                        # (1) 順伝播
    loss = F.cross_entropy(logits, Y[idx])        # (2) 損失
    optimizer.zero_grad()                         # (3) 勾配を消す
    loss.backward()                               # (4) 逆伝播
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # クリッピング
    optimizer.step()                              # (5) 更新
    # [end: standard]
    if step % 50 == 0 or step == MAX_STEPS - 1:
        print(f"ステップ {step:3d}  学習率 {lr:.5f}  損失 {loss.item():.4f}")

train_loss, train_acc = evaluate(model, X, Y)
test_loss, test_acc = evaluate(model, Xt, Yt)
n_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数 {n_params:,}")
print(f"訓練: 損失 {train_loss:.4f}  正解率 {train_acc * 100:.1f}%")
print(f"テスト: 損失 {test_loss:.4f}  正解率 {test_acc * 100:.1f}%")
print("---")

# 比較: 同じモデルを、SGD（学習率は一定、クリッピングなし）で学習させる
print(f"比較: 同じ 50 枚のモデル、ミニバッチ {BATCH} 件、{MAX_STEPS} ステップ")
print(f"標準セット          訓練 {train_acc * 100:5.1f}%"
      f"  テスト {test_acc * 100:5.1f}%")
for sgd_lr in (0.1, 0.01):
    torch.manual_seed(0)
    model = DeepMLP(50, init="he", norm=True, residual=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr)
    for step in range(MAX_STEPS):
        idx = torch.randint(0, len(X), (BATCH,))
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc, test = evaluate(model, X, Y)[1], evaluate(model, Xt, Yt)[1]
    print(f"SGD（学習率 {sgd_lr:<4g}）  訓練 {acc * 100:5.1f}%"
          f"  テスト {test * 100:5.1f}%")
