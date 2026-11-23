"""一般化アドバンテージ推定（GAE）。手計算の例で確かめる。"""
import torch


# [snippet: gae]
def gae(R, V, M, gamma, lam):
    """R, V, M: (T, n)。M[t]=1 は時刻 t がエピソードの内側であること。
    エピソードが終わった後の価値は 0 とする。"""
    T = R.shape[0]
    A = torch.zeros_like(R)
    last = torch.zeros(R.shape[1])               # Â_{t+1}
    for t in reversed(range(T)):
        m_next = M[t + 1] if t + 1 < T else torch.zeros_like(M[t])
        v_next = V[t + 1] * m_next if t + 1 < T else 0.0
        delta = R[t] + gamma * v_next - V[t]          # TD 誤差 δ_t
        last = delta + gamma * lam * m_next * last    # Â_t = δ_t + γλ Â_{t+1}
        A[t] = last * M[t]
    return A
# [end: gae]


if __name__ == "__main__":
    # [snippet: gae-example]
    R = torch.tensor([[0.0], [0.0], [1.0]])     # 最後にだけ報酬 1
    V = torch.tensor([[0.5], [0.6], [0.8]])     # 価値ネットの予測
    M = torch.ones(3, 1)
    gamma = 0.9
    for lam in [0.0, 0.5, 1.0]:
        A = gae(R, V, M, gamma, lam)
        print(f"λ={lam}: Â =", [round(a, 4) for a in A[:, 0].tolist()])
    G0 = 0 + gamma * 0 + gamma**2 * 1.0
    print("モンテカルロ: G_0 - V(s_0) =", round(G0 - 0.5, 4))
    # [end: gae-example]
