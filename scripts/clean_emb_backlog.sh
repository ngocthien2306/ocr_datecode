#!/bin/bash
# ============================================================================
# clean_emb_backlog.sh — Dọn 1 lần đống folder cũ trong ai_services/test_result.
#
# Giữ N folder emb_* mới nhất, xoá phần cũ. Khớp với retention runtime trong
# embedding_classifier.py (_trim_emb_debug_folders). Chạy 1 lần để xử lý backlog
# tồn đọng; sau đó AI service tự giữ ở mức N.
#
# Dùng:
#   ./scripts/clean_emb_backlog.sh                 # xoá (giữ newest 200)
#   DRY_RUN=1 ./scripts/clean_emb_backlog.sh       # CHỈ xem trước, không xoá
#   EMB_DEBUG_KEEP=100 ./scripts/clean_emb_backlog.sh
#   TEST_RESULT_DIR=/duong/dan/khac ./scripts/clean_emb_backlog.sh
# ============================================================================
set -uo pipefail

KEEP="${EMB_DEBUG_KEEP:-200}"
DIR="${TEST_RESULT_DIR:-$HOME/Source/ocr_datecode/ai_services/test_result}"
DRY_RUN="${DRY_RUN:-0}"

[ -d "$DIR" ] || { echo "❌ Không thấy thư mục: $DIR"; exit 1; }

# Liệt kê emb_* sắp theo timestamp ở cuối tên (emb_{serial}_{YYYYMMDD}_{HHMMSS}),
# CŨ NHẤT trước. Không stat từng folder.
mapfile -t SORTED < <(
    find "$DIR" -maxdepth 1 -type d -name 'emb_*' -printf '%f\n' 2>/dev/null \
    | awk -F_ '{print $(NF-1)"_"$NF"\t"$0}' | sort | cut -f2
)

TOTAL=${#SORTED[@]}
echo "📁 $DIR"
echo "   Tổng: $TOTAL folder emb_*  |  giữ newest: $KEEP  |  DRY_RUN=$DRY_RUN"

if (( TOTAL <= KEEP )); then
    echo "✅ Không có gì để xoá (≤ $KEEP)."
    exit 0
fi

DELN=$(( TOTAL - KEEP ))
echo "   Sẽ xoá $DELN folder cũ nhất..."
del=0
for ((i=0; i<DELN; i++)); do
    d="$DIR/${SORTED[$i]}"
    if [ "$DRY_RUN" = "1" ]; then
        echo "   [dry] ${SORTED[$i]}"
    else
        rm -rf -- "$d" && del=$((del+1))
    fi
done

if [ "$DRY_RUN" = "1" ]; then
    echo "ℹ️  Dry-run: không xoá gì. Bỏ DRY_RUN=1 để xoá thật."
else
    REMAIN=$(find "$DIR" -maxdepth 1 -type d -name 'emb_*' 2>/dev/null | wc -l)
    echo "✅ Đã xoá $del folder. Còn lại: $REMAIN"
fi
