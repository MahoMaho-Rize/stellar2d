#!/usr/bin/env bash
# 打包一個目錄成 tar.gz,上傳到 GitHub Release(unlimited,不進 git history)。
#
# 用法:
#   scripts/push_frames.sh <path_to_dir> [release_tag]
#
# 例:
#   scripts/push_frames.sh videos/frames_every100
#   scripts/push_frames.sh videos/frames_every100 assets-2026-05-02
#
# 若不給 tag,預設用當日 "assets-YYYY-MM-DD"。
# Release 不存在會自動 create;存在則 upload 到同一個 release(同名檔會 --clobber)。
# 上傳成功後印出 URL,並保留本地 tar.gz(如要清可加 --clean)。

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <directory> [release_tag] [--clean]" >&2
    exit 1
fi

SRC=""
TAG=""
CLEAN=0
for arg in "$@"; do
    if [[ "$arg" == "--clean" ]]; then
        CLEAN=1
    elif [[ -z "$SRC" ]]; then
        SRC="$arg"
    elif [[ -z "$TAG" ]]; then
        TAG="$arg"
    fi
done
[[ -z "$TAG" ]] && TAG="assets-$(date +%Y-%m-%d)"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: '$SRC' is not a directory" >&2
    exit 1
fi

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
BASENAME=$(basename "$SRC")
TARBALL="${BASENAME}.tar.gz"

echo "==> 打包 $SRC -> $TARBALL"
tar czf "$TARBALL" -C "$(dirname "$SRC")" "$BASENAME"
SIZE=$(du -h "$TARBALL" | cut -f1)
echo "    size: $SIZE"

echo "==> Release tag: $TAG @ $REPO"
if ! gh release view "$TAG" >/dev/null 2>&1; then
    echo "    建立新 release"
    gh release create "$TAG" \
        --title "Assets ($TAG)" \
        --notes "自動產出的幀/視頻/分析附件。由 scripts/push_frames.sh 上傳。"
fi

echo "==> 上傳 $TARBALL"
gh release upload "$TAG" "$TARBALL" --clobber

URL="https://github.com/$REPO/releases/download/$TAG/$TARBALL"
echo ""
echo "完成: $URL"
echo ""
echo "下載回來:"
echo "  gh release download $TAG -p '$TARBALL' && tar xzf $TARBALL"

if [[ $CLEAN -eq 1 ]]; then
    rm -f "$TARBALL"
    echo "(已清本地 $TARBALL)"
fi
