"""学習のとき（答えまで入れて 1 回で計算）と、推論のとき（1 トークンずつ
生成）とで、答えの各トークンに付く確率が同じになることを確かめる。

因果マスクがあるので、学習のときに答えの「4」「7」が入力に入っていても、
位置 9 の予測には位置 0〜9 しか使われない（カンニングにならない）。
実行:  python ch16/fig_train_vs_generate.py
出力:  ch16/out/fig_train_vs_generate.txt     CPU で数秒。
"""

import torch
import torch.nn.functional as F

from common import CKPT_DIR, OUT_DIR, device, tok
from llmbook.chat import encode_chat_example, render_chat
from llmbook.train import load_checkpoint

torch.manual_seed(0)
model = load_checkpoint(CKPT_DIR / "ch16_sft.pt", device).eval()
messages = [{"role": "user", "content": "12+35は？"},
            {"role": "assistant", "content": "47です。"}]
ids, labels = encode_chat_example(tok, messages)  # 入力 13 個と正解 13 個


def show(i: int) -> str:
    return tok.decode([i]).replace("<|end|>", "E")


lines = ["[学習のとき] 13 トークンを 1 回の順伝播に入れる"]
with torch.no_grad():
    logits, _ = model(torch.tensor([ids], device=device))  # (1, 13, V)
    probs = F.softmax(logits[0], dim=-1)  # (13, V): 13 か所の予測が一度に
train_p = {}
for t, y in enumerate(labels):
    if y != -100:  # 損失を計算する 4 か所
        train_p[t] = probs[t, y].item()
        lines.append(f"  位置 {t:2d} の予測（位置 0〜{t} を見る）: "
                     f"正解 {show(y)!r} の確率 {train_p[t]:.4f}")

lines.append("[推論のとき] <|assistant|> までを入れ、1 トークンずつ足す")
prompt = render_chat(messages[:1], add_generation_prompt=True)
idx = torch.tensor([tok.encode(prompt)], device=device)  # (1, 10)
end_id = tok.special_tokens["<|end|>"]
with torch.no_grad():
    for step in range(1, 6):
        p = F.softmax(model(idx)[0][0, -1], dim=-1)  # 最後の位置の予測だけ
        top = torch.topk(p, 3)
        nxt = top.indices[0].item()  # 温度 0: いちばん確率の高いもの
        cands = ", ".join(f"{show(i)!r} {v:.3f}" for v, i in
                          zip(top.values.tolist(), top.indices.tolist()))
        t = idx.size(1) - 1
        lines.append(f"  コマ {step}: 入力 {idx.size(1)} トークン -> "
                     f"上位 {cands} -> {show(nxt)!r} を末尾に足す"
                     f"（学習時の位置 {t} の確率 "
                     f"{train_p.get(t, float('nan')):.4f}）")
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == end_id:
            lines.append("  <|end|> が出たので止める")
            break
lines.append("生成した応答: "
             + repr(tok.decode(idx[0, len(tok.encode(prompt)):].tolist())))
(OUT_DIR / "fig_train_vs_generate.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
