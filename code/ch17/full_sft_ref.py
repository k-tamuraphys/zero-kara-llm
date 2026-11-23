"""比較対象: 第 16 章の全パラメータ SFT（sft.pt）を同じ物差しで測る。"""

from common import CKPT_DIR, accuracy, soseki_val_loss
from llmbook.train import load_checkpoint
from lora import count_params

model = load_checkpoint(CKPT_DIR / "sft.pt")
n_train, n_total = count_params(model)
print(f"学習するパラメータ: {n_train:,} / {n_total:,}")
print(f"ファイルサイズ: {(CKPT_DIR / 'sft.pt').stat().st_size / 1024:.0f} KB")
print(f"対話正答率（1,000 問）: {accuracy(model):.1%}")
print(f"漱石の検証損失: {soseki_val_loss(model):.3f}")
