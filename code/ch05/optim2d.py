"""第 5 章: 細長い谷で、オプティマイザの動きを比べる。

損失 L(w1, w2) = (w1^2 + 100 w2^2) / 2 の上を、(-4, 1) から 100 歩進む。
    cd code && python ch05/optim2d.py > ch05/out/optim2d.txt
軌跡は ch05/out/optim2d_<名前>.csv に書き出す。
"""
import torch

from ch05.optim import SGD, Adam, Momentum, RMSProp

OUT = "ch05/out"


# [snippet: valley]
def loss_fn(w):
    """細長い谷。w2 の向きは w1 の向きより 100 倍急。"""
    return 0.5 * (w[0] ** 2 + 100 * w[1] ** 2)


def run(make_optimizer, steps=100):
    w = torch.tensor([-4.0, 1.0], requires_grad=True)   # 出発点
    optimizer = make_optimizer([w])
    path = [(w[0].item(), w[1].item(), loss_fn(w).item())]
    for step in range(steps):
        loss = loss_fn(w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        path.append((w[0].item(), w[1].item(), loss_fn(w).item()))
    return path
# [end: valley]


def main():
    torch.set_num_threads(1)
    runs = [
        ("sgd_small", "SGD（学習率 0.005）", lambda p: SGD(p, lr=0.005)),
        ("sgd", "SGD（学習率 0.019）", lambda p: SGD(p, lr=0.019)),
        ("sgd_large", "SGD（学習率 0.021）", lambda p: SGD(p, lr=0.021)),
        ("momentum", "モーメンタム（学習率 0.005）",
         lambda p: Momentum(p, lr=0.005, beta=0.9)),
        ("rmsprop", "RMSProp（学習率 0.1）", lambda p: RMSProp(p, lr=0.1)),
        ("adam", "Adam（学習率 0.1）", lambda p: Adam(p, lr=0.1)),
    ]
    print("手法                            10 歩後     30 歩後    100 歩後")
    for key, name, make in runs:
        path = run(make)
        with open(f"{OUT}/optim2d_{key}.csv", "w") as f:
            f.write("step,w1,w2,loss\n")
            for i, (a, b, l) in enumerate(path):
                f.write(f"{i},{a:.6f},{b:.6f},{l:.6g}\n")
        pad = 28 - sum(2 if ord(c) > 255 else 1 for c in name)
        print(name + " " * pad + f"{path[10][2]:>12.4g}{path[30][2]:>12.4g}"
              f"{path[100][2]:>12.3g}")
    # Adam の最初の 1 歩を、バイアス補正の説明のために表示する
    w = torch.tensor([-4.0, 1.0], requires_grad=True)
    opt = Adam([w], lr=0.1)
    loss_fn(w).backward()
    g = w.grad.clone()
    opt.step()
    print("---")
    print("Adam の最初の 1 歩")
    print(f"勾配 g         = ({g[0]:.1f}, {g[1]:.1f})")
    print(f"m (補正前)     = ({opt.m[0][0]:.2f}, {opt.m[0][1]:.2f})")
    print(f"v (補正前)     = ({opt.v[0][0]:.4f}, {opt.v[0][1]:.4f})")
    print(f"動いた量       = ({w[0].item() + 4:.4f}, {w[1].item() - 1:.4f})")


if __name__ == "__main__":
    main()
