"""浮動小数点の形式（fp32 / fp16 / bf16）の性質と丸め誤差を確かめる。"""
import torch

torch.manual_seed(0)
torch.set_num_threads(2)
types = [torch.float32, torch.float16, torch.bfloat16]

# [snippet: formats]
for dt in types:
    fi = torch.finfo(dt)
    print(f"{str(dt)[6:]:8s} 最大 {fi.max:.3e}  最小正規 {fi.tiny:.3e}"
          f"  eps {fi.eps:.3e}")

for dt in types:
    one = torch.tensor(1.0, dtype=dt)
    a = (one + 1e-3).item()                       # 1.001 は表せるか
    b = torch.tensor(70000.0, dtype=dt).item()    # 7 万は表せるか
    c = torch.tensor(1e-6, dtype=dt).item()       # 小さな勾配は残るか
    print(f"{str(dt)[6:]:8s} 1+0.001 -> {a:.6f}  70000 -> {b:.6g}"
          f"  1e-6 -> {c:.3g}")
# [end: formats]

# [snippet: accumulate]
rows = []
for dt in types:
    s = torch.tensor(0.0, dtype=dt)
    for i in range(1, 10001):
        s = s + torch.tensor(1e-3, dtype=dt)   # 0.001 を 1 万回足す（正解 10）
        if i % 250 == 0:
            rows.append((i, str(dt)[6:], s.item()))
    print(f"{str(dt)[6:]:8s} 0.001 を 1 万回足すと {s.item():.4f}")
# [end: accumulate]
with open("ch12/out/accumulate.csv", "w") as f:
    f.write("i,float32,float16,bfloat16\n")
    for i in range(250, 10001, 250):
        v = {r[1]: r[2] for r in rows if r[0] == i}
        f.write(f"{i},{v['float32']:.5f},{v['float16']:.5f},{v['bfloat16']:.5f}\n")

# [snippet: matmul-error]
A = torch.randn(512, 512, dtype=torch.float64)
B = torch.randn(512, 512, dtype=torch.float64)
exact = A @ B
for dt in types:
    C = (A.to(dt) @ B.to(dt)).double()
    err = ((C - exact).norm() / exact.norm()).item()
    print(f"{str(dt)[6:]:8s} 512×512 の行列積の相対誤差 {err:.2e}")
# [end: matmul-error]
