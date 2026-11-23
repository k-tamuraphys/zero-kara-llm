#!/bin/bash
# PDF の指定ページを PNG にする（目視確認用）。使い方: tools/pdf2png.sh build/ch05_training.pdf 3 [dpi]
# 出力: build/preview/<名前>-<ページ>.png
source "$(dirname "$0")/env.sh"
cd "$BOOK"; mkdir -p build/preview
python - "$1" "$2" "${3:-110}" <<'PY'
import sys, pymupdf as fitz
pdf, page, dpi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
doc = fitz.open(pdf)
out = f"build/preview/{pdf.split('/')[-1][:-4]}-{page}.png"
doc[page - 1].get_pixmap(dpi=dpi).save(out)
print(out, f"({len(doc)} pages total)")
PY
