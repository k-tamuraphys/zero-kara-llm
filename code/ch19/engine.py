"""KV キャッシュ付きの生成エンジン。

llmbook の GPT.generate は、1 トークン作るたびに列全体を計算し直す。
ここでは同じ重みを使い、(1) 過去のキーとバリューを覚えておき、
(2) 複数の列を「スロット」に入れて同時に進める、生成器を作る。
各スロットは長さが違ってよく、いつでも空けたり入れ直したりできる。
"""

import math

import torch
import torch.nn.functional as F


class Engine:
    # [snippet: engine-init]
    def __init__(self, model, n_slots: int):
        cfg = model.config
        self.model = model.eval()
        self.n_slots, self.max_len = n_slots, cfg.block_size
        self.n_head, self.d = cfg.n_head, cfg.n_embd
        shape = (cfg.n_layer, n_slots, cfg.block_size,
                 cfg.n_head, cfg.n_embd // cfg.n_head)
        self.K = torch.zeros(shape)  # キーの置き場 (L, スロット, 位置, H, d/H)
        self.V = torch.zeros(shape)  # バリューの置き場
        # 各スロットに、いま何トークン入っているか
        self.length = torch.zeros(n_slots, dtype=torch.long)
    # [end: engine-init]

    # [snippet: engine-forward]
    @torch.no_grad()
    def forward(self, slots: torch.Tensor, idx: torch.Tensor,
                n_real: torch.Tensor | None = None) -> torch.Tensor:
        """スロット slots (B,) に新しいトークン idx (B, T) を足して計算する。

        各行の最後のトークンの位置のロジット (B, V) を返す。行ごとに長さが
        違うときは、右側を詰め物で埋め、本当の長さを n_real (B,) で渡す。
        """
        m, H, d = self.model, self.n_head, self.d
        B, T = idx.shape
        # 新しいトークンが入る位置。スロットごとに出発点が違う。
        pos = self.length[slots][:, None] + torch.arange(T)      # (B, T)
        x = m.tok_emb(idx) + m.pos_emb(pos)                      # (B, T, d)
        S = int(pos.max()) + 1  # 今回見る必要のあるキャッシュの長さ
        # 位置 p のトークンが見てよいのは、キャッシュの 0..p 番目だけ
        visible = torch.arange(S) <= pos[:, :, None]             # (B, T, S)
        rows = slots[:, None]                                    # (B, 1)
        for layer, block in enumerate(m.blocks):
            q, k, v = block.attn.qkv(block.ln1(x)).split(d, dim=2)
            # 新しいキーとバリューをキャッシュに書き込む
            self.K[layer, rows, pos] = k.view(B, T, H, d // H)
            self.V[layer, rows, pos] = v.view(B, T, H, d // H)
            q = q.view(B, T, H, d // H).transpose(1, 2)      # (B, H, T, d/H)
            keys = self.K[layer, slots, :S].transpose(1, 2)  # (B, H, S, d/H)
            vals = self.V[layer, slots, :S].transpose(1, 2)
            att = (q @ keys.transpose(-2, -1)) / math.sqrt(d // H)
            att = att.masked_fill(~visible[:, None], float("-inf"))
            y = F.softmax(att, dim=-1) @ vals                # (B, H, T, d/H)
            y = y.transpose(1, 2).reshape(B, T, d)
            x = x + block.attn.proj(y)
            x = x + block.mlp(block.ln2(x))
        if n_real is None:
            n_real = torch.full((B,), T)
        self.length[slots] += n_real  # 詰め物のぶんは、長さに数えない
        x_last = x[torch.arange(B), n_real - 1]                  # (B, d)
        return m.lm_head(m.ln_f(x_last))                         # (B, V)
    # [end: engine-forward]

    # [snippet: engine-prefill]
    def prefill(self, slots: torch.Tensor,
                prompts: list[list[int]]) -> torch.Tensor:
        """スロットを空にし、長さの違うプロンプトをまとめて読ませる。"""
        self.length[slots] = 0  # 長さを 0 に戻せば、古い中身は使われない
        n_real = torch.tensor([len(p) for p in prompts])
        idx = torch.zeros(len(prompts), int(n_real.max()), dtype=torch.long)
        for i, p in enumerate(prompts):  # 右側を詰め物（0 番）で埋める
            idx[i, :len(p)] = torch.tensor(p)
        return self.forward(slots, idx, n_real)
    # [end: engine-prefill]


# [snippet: pick]
def pick(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """ロジット (B, V) から次のトークン (B,) を選ぶ。温度 0 は貪欲法。"""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
# [end: pick]


# [snippet: generate-batch]
@torch.no_grad()
def generate_batch(engine: Engine, prompts: list[list[int]], stop_id: int,
                   max_new_tokens: int = 24) -> list[list[int]]:
    """複数のプロンプトの続きを同時に生成する（貪欲法）。長さは違ってよい。"""
    n = len(prompts)
    outs = [[] for _ in range(n)]
    # [snippet: batch-loop]
    # プロンプトをまとめて読ませ、それぞれの 1 個目のトークンを選ぶ
    last = pick(engine.prefill(torch.arange(n), prompts))
    active = torch.arange(n)  # まだ書いている途中のスロット
    for _ in range(max_new_tokens):
        for i in active.tolist():
            outs[i].append(int(last[i]))
        active = active[last[active] != stop_id]  # 書き終えた列を外す
        if len(active) == 0:
            break
        # 残っている列だけを、まとめて 1 トークン進める
        logits = engine.forward(active, last[active][:, None])
        last[active] = pick(logits)
    # [end: batch-loop]
    return outs
# [end: generate-batch]


def reply_batch(engine: Engine, tok, questions: list[str],
                max_new_tokens: int = 24) -> list[str]:
    """質問のリストに対する応答（<|end|> の手前まで）のリストを返す。"""
    from common import chat_prompt_ids
    end_id = tok.special_tokens["<|end|>"]
    prompts = [chat_prompt_ids(tok, q) for q in questions]
    outs = generate_batch(engine, prompts, end_id, max_new_tokens)
    return [tok.decode(o).split("<|end|>")[0] for o in outs]
