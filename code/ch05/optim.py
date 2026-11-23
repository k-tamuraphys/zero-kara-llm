"""第 5 章: オプティマイザを自作する（SGD、モーメンタム、Adam、AdamW）。

どのクラスも torch.optim と同じ使い方（zero_grad と step）ができる。
実行すると、同じモデルを torch.optim と自作の両方で学習させ、
パラメータの差を表示する。
    cd code && python ch05/optim.py > ch05/out/optim.txt
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# [snippet: sgd]
class SGD:
    """確率的勾配降下法。勾配の逆向きに lr 倍だけ動く。"""

    def __init__(self, params, lr):
        self.params = list(params)   # 更新するパラメータの一覧
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None            # 前のステップの勾配を捨てる

    @torch.no_grad()                 # 更新そのものは微分の対象にしない
    def step(self):
        for p in self.params:
            p -= self.lr * p.grad
# [end: sgd]


# [snippet: momentum]
class Momentum(SGD):
    """モーメンタム付き SGD。過去の勾配をためた「速度」v で動く。"""

    def __init__(self, params, lr, beta=0.9):
        super().__init__(params, lr)
        self.beta = beta
        self.v = [torch.zeros_like(p) for p in self.params]  # 速度は 0 から

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.v):
            v.mul_(self.beta).add_(p.grad)   # v ← βv + g
            p -= self.lr * v                 # w ← w − ηv
# [end: momentum]


class RMSProp(SGD):
    """勾配の 2 乗の移動平均で割る。Adam から「勢い」を除いたもの。"""

    def __init__(self, params, lr, beta2=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.beta2, self.eps = beta2, eps
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.v):
            g = p.grad
            v.mul_(self.beta2).add_((1 - self.beta2) * g * g)
            p -= self.lr * g / (v.sqrt() + self.eps)


# [snippet: adam]
class Adam(SGD):
    """Adam。勾配の平均 m で向きを決め、2 乗の平均 v で歩幅をそろえる。"""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay     # 0 より大きければ AdamW
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0                           # これまでのステップ数

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            p -= self.lr * self.weight_decay * p         # 重み減衰（AdamW）
            m.mul_(self.beta1).add_((1 - self.beta1) * g)      # 勾配の平均
            v.mul_(self.beta2).add_((1 - self.beta2) * g * g)  # 2 乗の平均
            m_hat = m / (1 - self.beta1 ** self.t)       # バイアス補正
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
# [end: adam]


def max_diff(model_a, model_b):
    """二つのモデルのパラメータの差の最大値。"""
    return max((a - b).abs().max().item()
               for a, b in zip(model_a.parameters(), model_b.parameters()))


def main():
    from ch04.data import make_spiral

    torch.set_num_threads(1)
    torch.manual_seed(0)
    X, Y = make_spiral(n_per_class=100, seed=0)
    X, Y = torch.tensor(X).double(), torch.tensor(Y)
    base = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32),
                         nn.ReLU(), nn.Linear(32, 3)).double()
    pairs = [
        ("SGD", lambda p: SGD(p, lr=0.1),
         lambda p: torch.optim.SGD(p, lr=0.1)),
        ("モーメンタム", lambda p: Momentum(p, lr=0.1, beta=0.9),
         lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
        ("Adam", lambda p: Adam(p, lr=0.01),
         lambda p: torch.optim.Adam(p, lr=0.01)),
        ("AdamW", lambda p: Adam(p, lr=0.01, weight_decay=0.1),
         lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)),
    ]
    print("200 ステップ学習したあとの、自作と torch.optim の差")
    for name, make_mine, make_torch in pairs:
        models, losses = [], []
        for make in (make_mine, make_torch):
            model = copy.deepcopy(base)          # 同じ初期値から始める
            optimizer = make(model.parameters())
            for step in range(200):
                loss = F.cross_entropy(model(X), Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            models.append(model)
            losses.append(loss.item())
        print(f"{name:<8s} 損失 {losses[0]:.6f} / {losses[1]:.6f}"
              f"  パラメータの差の最大 {max_diff(*models):.1e}")


if __name__ == "__main__":
    main()
