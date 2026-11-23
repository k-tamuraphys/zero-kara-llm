#!/bin/bash
# 1 章だけを組版する。使い方: tools/build_chapter.sh ch05_training
# 出力: build/ch05_training.pdf。 他章への参照は「??」になる（正常）。
set -u
source "$(dirname "$0")/env.sh"
cd "$BOOK"
name="$1"
[ -f "chapters/$name.tex" ] || { echo "chapters/$name.tex がない"; exit 1; }
python tools/extract_snippets.py
num=$(echo "$name" | sed -E 's/^ch0*([0-9]+)_.*/\1/')
pre=""
case "$name" in
  appA*) num=1; pre='\appendix';; appB*) num=2; pre='\appendix';;
  appC*) num=3; pre='\appendix';; appD*) num=4; pre='\appendix';;
  appE*) num=5; pre='\appendix';; appF*) num=6; pre='\appendix';;
  ans*) num=4; pre='\appendix';;
esac
cat > "build/$name.tex" <<TEX
\documentclass[b5paper,10pt,openany,oneside]{ltjsbook}
\input{preamble}
\begin{document}
\input{tools/xref_stub}
$pre
\setcounter{chapter}{$((num-1))}
\input{chapters/$name}
\bibliographystyle{plain}
\bibliography{refs}
\end{document}
TEX
run() { lualatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "build/$name.tex" > "build/$name.stdout" 2>&1; }
run || { echo "=== LaTeX エラー ==="; grep -n -A6 -E "^(\./|/).*:[0-9]+:|^!" "build/$name.stdout" | head -60; exit 1; }
if [ -f refs.bib ]; then (cd build && BIBINPUTS=..: bibtex "$name" > "$name.bibout" 2>&1); run; fi
run
echo "=== 警告 ==="
grep -E "Overfull \\\\hbox \(([0-9]{2,})" "build/$name.log" | head -20
grep -E "Citation .* undefined|Label .* multiply defined" "build/$name.log" | sort -u | head -20
echo "未定義の参照（他章へのものは可）:"; grep -oE "Reference \`[^']+' on page [0-9]+ undefined" "build/$name.log" | sort -u | head -40
grep -E "Missing character" "build/$name.log" | sort -u | head -10
python tools/extract_snippets.py 2>&1 | grep "$( echo "$name" | sed -E 's/^(ch[0-9]+)_.*/\1/')/" | head -20
pages=$(grep -oE "Output written on .*\(([0-9]+) pages" "build/$name.log" | grep -oE "[0-9]+ pages")
echo "=== OK: build/$name.pdf ($pages) ==="
