"""PPO（クリップ付き代理目的関数）で倒立振子を学習する。"""
import sys
import time
import torch
from reinforce import PolicyNet, value_net, rollout, summarize
from gae import gae

torch.set_num_threads(2)
OUT = "ch21/out"


# [snippet: ppo-loss]
def ppo_loss(policy, V, S, A, logp_old, adv, ret, clip,
             c_v=0.5, c_ent=0.01):
    dist = policy.dist(S)
    ratio = torch.exp(dist.log_prob(A) - logp_old)   # ρ_t = π_θ / π_θold
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    loss_pi = -torch.min(surr1, surr2).mean()        # クリップ付き代理目的
    loss_v = ((V(S).squeeze(-1) - ret) ** 2).mean()  # 価値ネットの回帰
    entropy = dist.entropy().mean()                  # 方策のエントロピー
    return loss_pi + c_v * loss_v - c_ent * entropy
# [end: ppo-loss]


# [snippet: ppo-diag]
@torch.no_grad()
def diagnostics(policy, S, A, logp_old, clip):
    """更新の後で、方策がどれだけ動いたかを測る。"""
    dist = policy.dist(S)
    ratio = torch.exp(dist.log_prob(A) - logp_old)
    kl = ((ratio - 1) - torch.log(ratio)).mean()      # KL(π_old‖π_θ) の推定
    clipped = ((ratio - 1).abs() > clip).float().mean()
    return kl.item(), clipped.item(), dist.entropy().mean().item()
# [end: ppo-diag]


# [snippet: ppo-train]
def train_ppo(seed, iters=80, n_ep=16, lr=1e-3, gamma=0.99, lam=0.95,
              clip=0.2, epochs=10, n_mb=4, log=None):
    torch.manual_seed(seed)
    policy, V = PolicyNet(), value_net()
    opt = torch.optim.Adam([*policy.parameters(), *V.parameters()], lr=lr)
    history = []
    for it in range(iters):
        S, A, R, M = rollout(policy, n_ep)                 # (T, n, ...)
        with torch.no_grad():
            logp_old = policy.dist(S).log_prob(A)          # 集めたときの確率
            adv = gae(R, V(S).squeeze(-1), M, gamma, lam)  # Â_t
            ret = adv + V(S).squeeze(-1)                   # V_φ の回帰目標
        keep = M.flatten().bool()                          # 有効な時刻だけ
        S, A, logp_old, adv, ret = (x.flatten(0, 1)[keep] for x in
                                    (S, A, logp_old, adv, ret))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)      # 正規化
        for ep in range(epochs):                           # 同じデータを再利用
            for idx in torch.randperm(len(A)).chunk(n_mb):  # ミニバッチ
                loss = ppo_loss(policy, V, S[idx], A[idx], logp_old[idx],
                                adv[idx], ret[idx], clip)
                opt.zero_grad(); loss.backward(); opt.step()
        history.append(M.sum(0).mean().item())
        if log is not None:
            log.append((it + 1, history[-1],
                        *diagnostics(policy, S, A, logp_old, clip)))
    return history, policy
# [end: ppo-train]


if __name__ == "__main__":
    t0 = time.time()
    runs, logs = [], []
    for seed in range(5):
        log = []
        hist, policy = train_ppo(seed, log=log)
        runs.append(hist); logs.append(log)
        if seed == 0:                        # gymnasium での評価に使う
            torch.save(policy.state_dict(), "checkpoints/ch21_ppo.pt")
    for it, L, kl, cf, ent in logs[0][9::10]:
        print(f"iter {it:3d}  平均長 {L:5.1f}  KL {kl:7.4f}"
              f"  クリップ率 {cf:.3f}  エントロピー {ent:.3f}")
    last = [sum(r[-10:]) / 10 for r in runs]
    print("最後の 10 反復の平均エピソード長（種ごと）",
          " ".join(f"{x:5.1f}" for x in last), f" 平均 {sum(last)/5:5.1f}")
    with open(f"{OUT}/ppo_diag.csv", "w") as f:
        f.write("iter,length,kl,clipfrac,entropy\n")
        for row in logs[0]:
            f.write(",".join(f"{v:.5g}" for v in row) + "\n")
    with open(f"{OUT}/ppo_curves.csv", "w") as f:
        f.write("episodes,ppo_mean,ppo_lo,ppo_hi\n")
        for i, (m, lo, hi) in enumerate(summarize(runs)):
            f.write(f"{(i + 1) * 16},{m:.1f},{lo:.1f},{hi:.1f}\n")
    print(f"所要時間 {time.time()-t0:.0f} 秒")
