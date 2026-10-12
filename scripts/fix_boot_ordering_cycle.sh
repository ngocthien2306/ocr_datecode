#!/usr/bin/env bash
#
# Sửa vòng phụ thuộc khiến ocr-all và ocr-backend không tự chạy lúc khởi động.
#
# TRIỆU CHỨNG
#   Tắt máy bật lại thì không có gì lên: không Firefox, không :8000, không :5173.
#   Gõ tay `sudo systemctl restart ocr-all` thì chạy bình thường.
#
# VÒNG NẰM Ở ĐÂU
#   1. multi-user.target  After= ocr-backend   (ngầm, do ocr-backend được enable)
#   2. ocr-backend        After= jtop          (drop-in 10-jtop.conf, thêm 20/08/2026)
#   3. jtop               After= multi-user.target   (lỗi có sẵn của jetson-stats)
#   Ba cạnh khép kín. systemd gỡ vòng bằng cách XOÁ job khởi động của
#   ocr-backend với ocr-all. Không unit nào `failed`, chỉ là không có gì chạy.
#
# CẮT CẠNH NÀO
#   Cạnh 3 là chỗ sai thật, nhưng KHÔNG cắt được bằng drop-in trên máy này:
#   systemd 249 âm thầm bỏ qua `After=` gán rỗng, đã thử và xác nhận trên
#   LineTine ngày 26/08. Bản reset đó chỉ chạy ở systemd mới hơn.
#
#   Nên cắt cạnh 2, tức bỏ `After=jtop.service` khỏi drop-in của chính chúng ta.
#   Giữ lại `Wants=jtop.service`: Wants không tạo thứ tự nên không sinh vòng,
#   mà vẫn đảm bảo jtop được kéo lên cùng backend.
#
#   Đổi lại: cuộc đua giữa backend và jtop quay lại. Chấp nhận được vì
#     - trước 20/08 các máy vẫn chạy đúng như vậy,
#     - mất số liệu phần cứng nhẹ hơn nhiều so với cả dây chuyền không khởi động,
#     - và `After=` vốn chưa bao giờ chữa được cuộc đua đó: jtop khai
#       `Type=simple`, nên After chỉ đảm bảo tiến trình đã fork, không đảm bảo
#       daemon đã sẵn sàng nhận kết nối.
#
#   Chữa gốc là cho vòng giám sát trong backend thử lại thay vì chết ở lần đầu
#   (`_monitoring_loop_jtop`, backend/app/services/jetson_monitoring_service.py).
#   Làm xong việc đó thì ràng buộc thứ tự này không còn cần thiết nữa.
#
# CÁCH DÙNG
#     sudo bash scripts/fix_boot_ordering_cycle.sh            # sửa và tự kiểm
#     sudo bash scripts/fix_boot_ordering_cycle.sh --undo     # trả lại như cũ
#
# Script KHÔNG tự bật dịch vụ. Bật lên là mở Firefox và cửa sổ kiểm camera trên
# màn hình máy đó, nên để người đứng cạnh máy quyết định.

set -uo pipefail

DROPIN=/etc/systemd/system/ocr-backend.service.d/10-jtop.conf
BACKUP=/etc/systemd/system/ocr-backend.service.d/10-jtop.conf.before-cycle-fix
# Lần sửa trước dùng cách gán After= rỗng cho jtop. Cách đó vô tác dụng trên
# systemd 249 nên dọn đi, để lại chỉ gây hiểu nhầm là đã sửa xong.
OLD_ATTEMPT=/etc/systemd/system/jtop.service.d/10-fix-ordering-cycle.conf

[ "$(id -u)" -eq 0 ] || { echo "Cần sudo: sudo bash $0 $*"; exit 1; }

echo "máy: $(hostname)"

if [ "${1:-}" = "--undo" ]; then
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$DROPIN"
    systemctl daemon-reload
    echo "  đã trả lại drop-in cũ. Lưu ý: vòng phụ thuộc quay lại."
  else
    echo "  không có bản lưu, không có gì để trả lại"
  fi
  exit 0
fi

if [ -f "$OLD_ATTEMPT" ]; then
  rm -f "$OLD_ATTEMPT"
  rmdir /etc/systemd/system/jtop.service.d 2>/dev/null
  # Nạp lại NGAY: xoá file mà không reload thì systemd vẫn giữ đường dẫn drop-in
  # trong bộ nhớ, và `systemctl cat jtop` trả lỗi vì đọc một file không còn nữa.
  systemctl daemon-reload
  echo "  đã dọn bản sửa jtop cũ (vô tác dụng trên systemd $(systemctl --version | head -1 | awk '{print $2}'))"
fi

if [ ! -f "$DROPIN" ]; then
  echo "  máy này không có $DROPIN, không dính vòng"
else
  echo "  ocr-backend After (trước): $(systemctl show ocr-backend -p After --value | tr ' ' '\n' | grep -c jtop) tham chiếu jtop"
  [ -f "$BACKUP" ] || cp "$DROPIN" "$BACKUP"
  cat > "$DROPIN" <<'CONF'
# Backend đọc số liệu phần cứng qua jtop, nên vẫn kéo jtop lên cùng.
#
# ĐÃ BỎ `After=jtop.service` ngày 26/08/2026. Ràng buộc thứ tự đó khép kín một
# vòng phụ thuộc (multi-user.target -> ocr-backend -> jtop -> multi-user.target,
# vì jtop.service của jetson-stats tự khai After=multi-user.target), và systemd
# gỡ vòng bằng cách xoá job khởi động của ocr-backend với ocr-all: sau mỗi lần
# bật máy thì không có gì chạy.
#
# Bỏ After không làm mất gì thật sự: jtop khai Type=simple nên After chỉ đảm bảo
# tiến trình đã fork, chưa bao giờ đảm bảo daemon sẵn sàng. Chữa đúng chỗ là cho
# _monitoring_loop_jtop trong backend thử lại thay vì chết ở lần nối đầu tiên.
#
# Wants không tạo thứ tự nên không sinh vòng.
[Unit]
Wants=jtop.service
CONF
  systemctl daemon-reload
  echo "  ocr-backend After (sau) : $(systemctl show ocr-backend -p After --value | tr ' ' '\n' | grep -c jtop) tham chiếu jtop"
fi

echo
echo "  --- kiểm tra còn vòng không ---"
OUT=$(systemd-analyze verify multi-user.target 2>&1 | grep -i "ordering cycle")
if [ -n "$OUT" ]; then
  echo "  VẪN CÒN VÒNG:"; echo "$OUT" | sed 's/^/    /'
  echo "  Chưa sửa xong, đừng khởi động lại máy khi chưa xử lý tiếp."
  exit 1
fi
echo "  không còn vòng nào"

echo
echo "  --- trạng thái hiện tại ---"
for u in ocr-backend ocr-all ocr-agent jtop; do
  printf "    %-14s %s / %s\n" "$u" \
    "$(systemctl is-active  "$u" 2>/dev/null)" \
    "$(systemctl is-enabled "$u" 2>/dev/null)"
done

echo
echo "  Bản sửa có hiệu lực từ lần khởi động máy kế tiếp."
echo "  Gỡ bản sửa: sudo bash $0 --undo"
