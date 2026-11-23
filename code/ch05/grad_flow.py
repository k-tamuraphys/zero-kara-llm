"""第 5 章: 隠れ層 50 枚の MLP で、学習前の勾配の大きさを層ごとに測る。

    cd code && python ch05/grad_flow.py > ch05/out/grad_flow.txt
"""
import torch
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, load_spiral

torch.set_num_threads(1)
X, Y, _, _ = load_spiral()
DEPTH = 50


def grad_norms(**kw):
    """学習前に 1 回だけ逆伝播し、各層の重みの勾配のノルムを返す。"""
    torch.manual_seed(0)
    model = DeepMLP(DEPTH, **kw)
    loss = F.cross_entropy(model(X), Y)
    loss.backward()
    layers = [model.first, *model.layers]
    return [layer.weight.grad.norm().item() for layer in layers]


cols = {
    "default": grad_norms(init="default"),
    "he": grad_norms(init="he"),
    "he_ln": grad_norms(init="he", norm=True),
    "he_res_ln": grad_norms(init="he", norm=True, residual=True),
}
with open("ch05/out/grad_flow.csv", "w") as f:
    f.write("layer," + ",".join(cols) + "\n")
    for i in range(DEPTH):
        f.write(f"{i + 1}," + ",".join(f"{v[i]:.4e}" for v in cols.values())
                + "\n")
names = {"default": "PyTorch の標準", "he": "He", "he_ln": "He＋正規化",
         "he_res_ln": "He＋正規化＋残差"}
print("隠れ層 50 枚、学習前。各層の重みの勾配の大きさ")
print("                    第 1 層    第 25 層    第 50 層")
for key, v in cols.items():
    pad = 18 - sum(2 if ord(c) > 255 else 1 for c in names[key])
    print(names[key] + " " * pad + f"{v[0]:10.1e} {v[24]:10.1e} {v[49]:10.1e}")
