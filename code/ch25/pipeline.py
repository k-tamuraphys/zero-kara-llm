"""第 25 章：LLM 開発の全工程を 1 本で通す。

  python ch25/pipeline.py mini [テキストファイル]   # 縮小版を最初から（約 3 分）
  python ch25/pipeline.py eval 名前                  # 既存のチェックポイントを測り直す
     名前: base / sft / lora / int8 / rl / agent
出力: ch25/out/mini.txt, mini_pretrain.csv, mini_rl.csv, eval_名前.txt
"""
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from llmbook.corpus import DATA_DIR, load_corpus
from llmbook.tokenizer import BPETokenizer
from llmbook.model import GPT, GPTConfig
from llmbook.train import train_lm, load_checkpoint, make_optimizer
from llmbook.arith import drill_text, drill_accuracy
from llmbook.chat import (make_chat_example, encode_chat_example, collate,
                          render_chat, chat_accuracy)

torch.set_num_threads(2)
OUT = HERE / "out"
CKPT = DATA_DIR.parent / "checkpoints"
log_file = None


def log(*a):
    print(*a, flush=True)
    print(*a, file=log_file, flush=True)


# [snippet: config]
CFG = dict(
    n_chars=200_000,     # コーパスの先頭から使う字数（本物: 数兆トークン）
    n_drill=20_000,      # 混ぜる計算ドリルの行数
    vocab=512,           # BPE の語彙（本物: 3 万〜26 万）
    model=GPTConfig(vocab_size=512, block_size=64, n_layer=2,
                    n_head=2, n_embd=96),
    pre_steps=600,       # 事前学習のステップ数
    sft_steps=1000,      # SFT のステップ数
    rl_steps=40,         # 強化学習（GRPO）のステップ数
    max_value=99,        # 問題は 2 桁まで（小さなモデルに合わせる）
)
# [end: config]


# [snippet: stage-data]
def stage_data(path):
    """1. データ: 文章とドリルを混ぜ、訓練と検証に分ける。"""
    text = Path(path).read_text() if path else load_corpus()
    text = text[:CFG["n_chars"]]
    drill = drill_text(CFG["n_drill"], seed=0, max_value=CFG["max_value"])
    lines = (text + "\n" + drill).split("\n")
    random.Random(0).shuffle(lines)        # 文章とドリルを行ごとに混ぜる
    mixed = "\n".join(lines)
    cut = int(len(mixed) * 0.95)
    return mixed[:cut], mixed[cut:]
# [end: stage-data]


# [snippet: stage-tokenizer]
def stage_tokenizer(train_text):
    """2. トークナイザ: 自分のデータで BPE を学習する。"""
    tok = BPETokenizer()
    tok.train(train_text[:100_000], CFG["vocab"])   # 先頭 10 万字で学習
    return tok
# [end: stage-tokenizer]


# [snippet: stage-pretrain]
def stage_pretrain(tok, train_text, val_text):
    """3. 事前学習: 次のトークンを当てる（第 10・13 章の train_lm）。"""
    tr = torch.tensor(tok.encode(train_text))
    va = torch.tensor(tok.encode(val_text))
    torch.manual_seed(0)
    model = GPT(CFG["model"])
    train_lm(model, tr, va, max_steps=CFG["pre_steps"], batch_size=32,
             max_lr=3e-3, warmup_steps=50, eval_interval=100,
             log_path=OUT / "mini_pretrain.csv")
    return model, len(tr)
# [end: stage-pretrain]


# [snippet: stage-sft]
def stage_sft(model, tok):
    """4. SFT: 対話形式の手本で、応答の部分だけを学ぶ（第 16 章）。"""
    rng = random.Random(1)
    data = [encode_chat_example(tok, make_chat_example(
        rng, CFG["max_value"])[0]) for _ in range(2000)]
    opt = make_optimizer(model, lr=1e-3)
    model.train()
    for step in range(CFG["sft_steps"]):
        x, y = collate(random.Random(step).sample(data, 32),
                       tok.special_tokens["<|pad|>"])
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
# [end: stage-sft]


# [snippet: stage-rl]
def stage_rl(model, tok, Q=8, G=8):
    """5. 強化学習: 正誤だけを報酬に GRPO で鍛える（第 22 章の縮小版）。"""
    end = tok.special_tokens["<|end|>"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng, rows = random.Random(2), []
    for step in range(CFG["rl_steps"]):
        loss, rs = 0.0, []
        for _ in range(Q):                         # Q 問 × G 回答
            msgs, ans = make_chat_example(rng, CFG["max_value"])
            p = tok.encode(render_chat(msgs[:1], add_generation_prompt=True))
            idx = torch.tensor([p] * G)
            model.eval()
            with torch.no_grad():
                out = model.generate(idx, 8, temperature=1.0)
            gen = out[:, len(p):]                  # (G, 8) 生成部分
            mask = (gen == end).long().cumsum(1)   # <|end|> の後は 0 に
            mask = ((mask == 0) | ((mask == 1) & (gen == end))).float()
            r = torch.tensor([float(tok.decode(g.tolist()).split(
                "<|end|>")[0] == f"{ans}です。") for g in gen])
            A = (r - r.mean()) / (r.std() + 1e-4)  # グループ内で正規化
            model.train()
            logits, _ = model(out[:, :-1])
            lp = F.log_softmax(logits[:, len(p) - 1:], -1)
            lp = lp.gather(2, gen[..., None]).squeeze(2)  # (G, 8)
            loss = loss - (A[:, None] * lp * mask).sum() / mask.sum() / Q
            rs.append(r.mean().item())
        opt.zero_grad(); loss.backward(); opt.step()
        rows.append((step, sum(rs) / Q))
    return rows
# [end: stage-rl]


# [snippet: stage-quant]
def stage_quant(model):
    """6. 量子化: 2 次元の重みを 8 ビット整数に丸めて戻す（第 18 章）。"""
    import copy
    m = copy.deepcopy(model)
    for p in m.parameters():
        if p.dim() == 2:                           # 行ごとに最小〜最大を 255 等分
            lo = p.data.min(1, keepdim=True).values
            hi = p.data.max(1, keepdim=True).values
            s = (hi - lo).clamp(min=1e-8) / 255
            p.data = torch.round((p.data - lo) / s) * s + lo
    return m
# [end: stage-quant]


def acc(model, tok, n=200):
    return chat_accuracy(model, tok, n_problems=n, seed=12345,
                         max_value=CFG["max_value"])


def run_mini(path):
    t0 = time.time()
    tt = lambda: f"{time.time() - t0:6.1f} 秒"
    tr_text, va_text = stage_data(path)
    log(f"[1 データ]       {len(tr_text):,} 字（検証 {len(va_text):,} 字）  {tt()}")
    tok = stage_tokenizer(tr_text)
    cpt = len(tr_text[:20000]) / len(tok.encode(tr_text[:20000]))
    log(f"[2 トークナイザ] 語彙 {tok.vocab_size}、{cpt:.2f} 字/トークン  {tt()}")
    model, n_tok = stage_pretrain(tok, tr_text, va_text)
    import csv
    last = list(csv.DictReader(open(OUT / "mini_pretrain.csv")))[-1]
    log(f"[3 事前学習]     {model.num_params():,} パラメータ、{n_tok:,} トークン、"
        f"検証損失 {float(last['val_loss']):.3f}  {tt()}")
    log(f"                 対話の正答率 {acc(model, tok):.1%}  {tt()}")
    l = stage_sft(model, tok)
    a_sft = acc(model, tok)
    log(f"[4 SFT]          損失 {l:.3f}、対話の正答率 {a_sft:.1%}  {tt()}")
    rows = stage_rl(model, tok)
    with open(OUT / "mini_rl.csv", "w") as f:
        f.write("step,reward\n")
        f.writelines(f"{s},{r:.4f}\n" for s, r in rows)
    log(f"[5 強化学習]     {len(rows)} ステップ、対話の正答率 {acc(model, tok):.1%}"
        f"  {tt()}")
    q = stage_quant(model)
    log(f"[6 量子化 int8]  対話の正答率 {acc(q, tok):.1%}  {tt()}")


def run_eval(name):
    """既存の成果物を、各章と同じ物差しで測り直す。"""
    tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
    t0 = time.time()
    if name == "base":
        m = load_checkpoint(CKPT / "base.pt")
        log(f"base.pt ドリル形式 200 問: {drill_accuracy(m, tok):.1%}")
    elif name in ("sft", "rl", "int8", "lora"):
        m = load_checkpoint(CKPT / ("ch22_rl.pt" if name == "rl" else "sft.pt"))
        if name == "int8":
            m = stage_quant(m)
        if name == "lora":
            sys.path.insert(0, str(HERE.parent / "ch17"))
            from lora import add_lora
            m = add_lora(load_checkpoint(CKPT / "base.pt"), r=8, alpha=16.0)
            m.load_state_dict(torch.load(CKPT / "ch17_lora_r8_all.pt"),
                              strict=False)
        a = chat_accuracy(m, tok, n_problems=1000, seed=777)
        log(f"{name}: 対話 1,000 問（種 777）: {a:.1%}")
    elif name == "agent":
        sys.path.insert(0, str(HERE.parent / "ch24"))
        import env
        m = load_checkpoint(CKPT / "ch24_agentic_rl.pt")
        res, _ = env.evaluate(m, env.eval_sets())
        for k, v in res.items():
            log(f"agent {k}: 正答率 {v['acc']:.1%}、電卓 {v['tool']:.1%}")
    log(f"所要 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    log_file = open(OUT / (f"eval_{arg}.txt" if cmd == "eval" else "mini.txt"),
                    "w")
    run_mini(arg) if cmd == "mini" else run_eval(arg)
