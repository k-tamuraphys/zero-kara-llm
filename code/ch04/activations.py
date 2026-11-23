"""4 つの活性化関数（ReLU, sigmoid, tanh, GELU）の値と傾きを保存する。"""

import torch
import torch.nn.functional as F

ACTS = {"relu": torch.relu, "sigmoid": torch.sigmoid,
        "tanh": torch.tanh, "gelu": F.gelu}


def value_and_slope(act, x):
    """各点での関数の値と、傾き（導関数の値。autograd で求める）を返す。"""
    x = x.clone().requires_grad_(True)
    y = act(x)
    y.sum().backward()
    return y.detach(), x.grad


x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print("入力 u       " + "".join(f"{v:8.1f}" for v in x.tolist()))
for name, act in ACTS.items():
    y, dy = value_and_slope(act, x)
    print(f"{name:8s} 値  " + "".join(f"{v:8.3f}" for v in y.tolist()))
    print(f"{'':8s} 傾き" + "".join(f"{v:8.3f}" for v in dy.tolist()))

xs = torch.linspace(-4.0, 4.0, 161)
cols = {"u": xs}
for name, act in ACTS.items():
    cols[name], cols["d" + name] = value_and_slope(act, xs)
with open("ch04/out/activations.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for i in range(len(xs)):
        f.write(",".join(f"{v[i].item():.5f}" for v in cols.values()) + "\n")
