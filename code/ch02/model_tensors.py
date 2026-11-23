"""基盤モデルの正体を見る: パラメータは名前の付いたテンソルの集まり。"""

from llmbook.corpus import DATA_DIR
from llmbook.train import load_checkpoint

# [snippet: list]
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
params = list(model.named_parameters())  # (名前, テンソル) の組のリスト
for name, p in params[:5]:  # 先頭の 5 個だけ表示する
    print(f"{name:26} {str(tuple(p.shape)):12} {p.dtype}")
print("...")
total = sum(p.numel() for name, p in params)
print(f"テンソルの数 {len(params)}、要素数の合計 {total:,}")
print(f"メモリ {total * 4 / 1e6:.1f} MB（float32 は 1 要素 4 バイト）")
# [end: list]
