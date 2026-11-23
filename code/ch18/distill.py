"""第 18 章: 知識蒸留の部品（温度付き softmax、蒸留損失、教師による生成）。

python ch18/distill.py で、教師の「ダークナレッジ」と τ² の効果を表示する。
"""
import random

import torch
import torch.nn.functional as F

from llmbook.chat import IGNORE, make_chat_example, render_chat


# [snippet: softmax-tau]
def softmax_tau(z: torch.Tensor, tau: float) -> torch.Tensor:
    """温度 tau 付き softmax。tau > 1 で分布がなだらかになる。"""
    return F.softmax(z / tau, dim=-1)
# [end: softmax-tau]


# [snippet: distill-loss]
def distill_loss(z_s, z_t, y, tau: float = 2.0, alpha: float = 0.5):
    """alpha * tau^2 * KL(教師 || 生徒) + (1 - alpha) * CE(正解, 生徒)。

    z_s, z_t: 生徒と教師のロジット (B, T, V)。y: 正解 (B, T)、-100 は無視。
    """
    mask = y != IGNORE                      # 答えの位置だけ使う (B, T)
    zs, zt = z_s[mask], z_t[mask]           # (N, V)。N は答えの位置の数
    log_ps = F.log_softmax(zs / tau, dim=-1)
    log_pt = F.log_softmax(zt / tau, dim=-1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1).mean()
    ce = F.cross_entropy(zs, y[mask])       # 温度 1 で正解と比べる
    return alpha * tau**2 * kl + (1 - alpha) * ce, kl, ce
# [end: distill-loss]


# [snippet: teacher-generate]
@torch.no_grad()
def teacher_answers(teacher, tok, questions: list[str],
                    max_new: int = 8) -> list[str]:
    """教師に貪欲生成で答えさせる（長さが同じ質問をまとめて処理）。"""
    teacher.eval()
    end_id = tok.special_tokens["<|end|>"]
    prompts = [tok.encode(render_chat([{"role": "user", "content": q}],
                                      add_generation_prompt=True))
               for q in questions]
    replies = [""] * len(questions)
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    for idxs in by_len.values():
        x = torch.tensor([prompts[i] for i in idxs])      # (b, T)
        for _ in range(max_new):
            logits, _ = teacher(x)
            nxt = logits[:, -1].argmax(-1, keepdim=True)  # (b, 1)
            x = torch.cat([x, nxt], dim=1)
        for row, i in zip(x.tolist(), idxs):
            gen = row[len(prompts[i]):]
            if end_id in gen:
                gen = gen[:gen.index(end_id)]
            replies[i] = tok.decode(gen)
    return replies
# [end: teacher-generate]


def main():
    from llmbook.corpus import DATA_DIR
    from llmbook.tokenizer import BPETokenizer
    from llmbook.train import load_checkpoint
    torch.set_num_threads(2)
    tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
    teacher = load_checkpoint(DATA_DIR.parent / "checkpoints" / "sft.pt")
    teacher.eval()
    # [snippet: dark-knowledge]
    prompt = render_chat([{"role": "user", "content": "12+35は？"}],
                         add_generation_prompt=True)
    x = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        z = teacher(x)[0][0, -1]                 # 答えの 1 文字目のロジット
    for tau in (1.0, 2.0, 4.0):
        p = softmax_tau(z, tau)
        top = torch.topk(p, 5)
        cells = [f"{tok.decode([i])!r}:{v:.3f}"
                 for v, i in zip(top.values.tolist(), top.indices.tolist())]
        print(f"tau={tau}: " + "  ".join(cells))
    # [end: dark-knowledge]
    # τ を変えたときの、KL 項の勾配の大きさ（τ² を掛けない場合と掛けた場合）
    print("\ntau  |grad KL|  |grad tau^2*KL|")
    torch.manual_seed(0)
    z_t = torch.randn(64, 4096) * 3
    for tau in (1.0, 2.0, 4.0, 8.0):
        z_s = (torch.randn(64, 4096) * 3).requires_grad_()
        y = torch.zeros(64, dtype=torch.long)
        _, kl, _ = distill_loss(z_s[None], z_t[None], y[None], tau, 1.0)
        g, = torch.autograd.grad(kl, z_s)
        print(f"{tau:3.0f}  {g.norm():9.5f}  {tau**2 * g.norm():13.5f}")
    rng = random.Random(0)
    qs = [make_chat_example(rng)[0][0]["content"] for _ in range(4)]
    for q, a in zip(qs, teacher_answers(teacher, tok, qs)):
        print(f"{q} -> {a}")


if __name__ == "__main__":
    main()
