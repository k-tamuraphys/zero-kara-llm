#!/bin/bash
# 本全体を組版する。出力: build/main.pdf
set -u
source "$(dirname "$0")/env.sh"
cd "$BOOK"
python tools/extract_snippets.py
run() { lualatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex > build/main.stdout 2>&1; }
run || { echo "=== LaTeX エラー ==="; grep -n -A6 -E "^(\./|/).*:[0-9]+:|^!" build/main.stdout | head -60; exit 1; }
(cd build && BIBINPUTS=..: bibtex main > main.bibout 2>&1; upmendex -q main.idx 2>/dev/null || makeindex -q main.idx)
run; run
grep -cE "Overfull \\\\hbox \(([0-9]{2,})" build/main.log
grep -E "undefined|multiply defined" build/main.log | sort -u | head -40
grep -oE "Output written on .*" build/main.log
