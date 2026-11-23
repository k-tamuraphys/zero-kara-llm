"""コイン投げの最尤推定: 尤度と対数尤度を計算し、最大の場所を確かめる。"""

import numpy as np

# [snippet: likelihood]
n, k = 10, 7  # 10 回投げて表が 7 回
theta = np.linspace(0.01, 0.99, 99)  # θ の候補: 0.01, 0.02, ..., 0.99
lik = theta**k * (1 - theta) ** (n - k)  # 尤度 p(D|θ)
loglik = k * np.log(theta) + (n - k) * np.log(1 - theta)  # 対数尤度
print(f"尤度が最大の θ     = {theta[np.argmax(lik)]:.2f}")
print(f"対数尤度が最大の θ = {theta[np.argmax(loglik)]:.2f}")
for t in [0.5, 0.7, 0.9]:
    i = np.argmin(np.abs(theta - t))  # t に最も近い候補の番号
    print(f"θ = {t}: 尤度 = {lik[i]:.6f}, 対数尤度 = {loglik[i]:.3f}")
# [end: likelihood]

with open("ch01/out/coin.csv", "w") as fp:  # 図用
    fp.write("theta,lik,loglik\n")
    for t, a, b in zip(theta, lik, loglik):
        fp.write(f"{t:.2f},{a:.6e},{b:.4f}\n")

print("---")
# [snippet: underflow]
n, k = 10000, 7000  # 1 万回投げて表が 7000 回
lik = 0.7**k * 0.3 ** (n - k)  # とても小さな数を 1 万個掛ける
loglik = k * np.log(0.7) + (n - k) * np.log(0.3)  # 対数なら足し算
print("尤度     =", lik)
print("対数尤度 =", loglik)
# [end: underflow]
