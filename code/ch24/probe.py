"""学習前の ch23_tool_sft.pt を調べる：質問の型ごとの P(最初が <|tool|>)、
貪欲での正答率・電卓使用率、強制的に電卓を使わせたときの正答率。"""
import re, time
import torch, torch.nn.functional as F
from env import *

torch.set_num_threads(2)
t0 = time.time()
model = load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt")
sets = eval_sets()
def qtype(q):
    if "足すと" in q or "引くと" in q: return "「aとbを足すと/aからbを引くと…ですか？」"
    if "足す" in q or "引く" in q: return "「a足すbは？/a引くbは？」"
    return "「a+bは？/a-bは？」"
with torch.no_grad():
    groups = {}
    for t in sets["add"]:
        ids = torch.tensor([tok.encode(f"<|user|>{t.question}<|end|>")])
        p = F.softmax(model(ids)[0][0, -1], -1)[TOOL].item()
        groups.setdefault(qtype(t.question), []).append(p)
print("加減算 1,000 問：最初のトークンが <|tool|> になる確率（温度 1）")
for k, v in groups.items():
    print(f"  {k:28s} {len(v):4d} 問  平均 {sum(v)/len(v):.3f}")
res, trajs = evaluate(model, sets)
print("貪欲（温度 0）での成績")
for k, r in res.items():
    print(f"  {k}: 正答率 {r['acc']:.3f}  電卓使用率 {r['tool']:.3f}  平均呼び出し {r['calls']:.2f}")
# 最初を <|tool|> に強制（eps=1 で半々 → ここでは強制用に直接）
import env
forced = []
for s in range(0, 1000, 250):
    forced += rollout(model, sets["add"][s:s+250], temperature=0.0, eps=1.0,
                      gen=torch.Generator().manual_seed(0))
tool_tr = [t for t in forced if t.n_calls > 0]
no_tr = [t for t in forced if t.n_calls == 0]
print(f"加減算で最初の役割トークンを半々に強制：電卓あり {len(tool_tr)} 問の正答率 "
      f"{sum(reward(t,0) for t in tool_tr)/len(tool_tr):.3f}、電卓なし {len(no_tr)} 問の正答率 "
      f"{sum(reward(t,0) for t in no_tr)/len(no_tr):.3f}")
print("強制した軌跡の例:")
for t in tool_tr[:3]: print("  ", t.task.question, "".join(t.texts))
print(f"[{time.time()-t0:.0f} 秒]")
