# 使い方: source tools/env.sh
# 本を組版するための設定。LuaLaTeX（TeX Live）と Python 3 が PATH にあれば、このままで動く。
# 置き場所が特殊な環境では、tools/env.local.sh を作り、そこで PATH などを足す。
_tools="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BOOK="$(dirname "$_tools")"
export PYTHONPATH=$BOOK/code${PYTHONPATH:+:$PYTHONPATH}
[ -f "$_tools/env.local.sh" ] && source "$_tools/env.local.sh"
unset _tools
