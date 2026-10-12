#!/usr/bin/env bash
#
# Chạy TRÊN MÁY MAC. Sửa vòng phụ thuộc systemd cho cả đội hình.
#
# Xem `scripts/fix_boot_ordering_cycle.sh` để biết vì sao có vòng. Tóm tắt:
# `jtop.service` của gói jetson-stats vừa khai `WantedBy=multi-user.target` vừa
# khai `After=multi-user.target`. Cặp này khép vòng khi có unit nào đứng giữa,
# và systemd gỡ vòng bằng cách XOÁ job khởi động của ocr-backend với ocr-all.
# Không unit nào `failed`, chỉ là sau khi bật máy thì không có gì chạy.
#
# CÁCH DÙNG
#
#   bash scripts/fix_boot_cycle_fleet.sh --check      # chỉ xem, không cần sudo
#   bash scripts/fix_boot_cycle_fleet.sh              # sửa, mỗi máy hỏi mật khẩu sudo một lần
#   bash scripts/fix_boot_cycle_fleet.sh --start      # sửa, và bật ocr-all ở máy đang tắt
#   bash scripts/fix_boot_cycle_fleet.sh --print      # in lệnh ra để tự dán từng máy
#   bash scripts/fix_boot_cycle_fleet.sh --undo       # gỡ bản sửa
#   bash scripts/fix_boot_cycle_fleet.sh M2 LineTine  # chỉ vài máy
#
# Mỗi máy sẽ hỏi mật khẩu sudo một lần, gõ ngay trên cửa sổ terminal. Vì thế
# script này PHẢI chạy trong terminal thật, không chạy nền được.
#
# `--start` bật ocr-all ở máy nào đang tắt. Mặc định TẮT, vì bật là mở Firefox
# và cửa sổ kiểm camera trên màn hình máy đó, nên phải có người biết trước.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT="$HERE/fix_boot_ordering_cycle.sh"
REMOTE=/tmp/fix_boot_ordering_cycle.sh

# PC-Auto-1 KHÔNG nằm trong danh sách: máy đó chạy nhánh release_v2, quản lý
# riêng. Nó cũng là máy x86 nên không có jtop, không dính vòng này. Muốn chạy
# cho nó thì truyền tên vào: `bash ... PC-Auto-1`.
HOSTS_DEFAULT=(Auto2 M1 M2 LineTine Tin-2)

MODE=fix
START=0
HOSTS=()
for a in "$@"; do
  case "$a" in
    --check) MODE=check ;;
    --print) MODE=print ;;
    --undo)  MODE=undo ;;
    --start) START=1 ;;
    -*) echo "tham số lạ: $a"; exit 2 ;;
    *)  HOSTS+=("$a") ;;
  esac
done
[ ${#HOSTS[@]} -eq 0 ] && HOSTS=("${HOSTS_DEFAULT[@]}")

[ -f "$AGENT" ] || { echo "thiếu $AGENT"; exit 1; }

SSH=(ssh -o ConnectTimeout=10 -o ServerAliveInterval=5)

# Trạng thái một máy, đọc thôi, không cần sudo.
probe() {
  "${SSH[@]}" -o BatchMode=yes "$1" '
    has_jtop=$(systemctl cat jtop.service >/dev/null 2>&1 && echo có || echo không)
    dp=/etc/systemd/system/ocr-backend.service.d/10-jtop.conf
    if [ ! -f "$dp" ]; then fixed="n/a"
    elif grep -q "^After=jtop" "$dp"; then fixed=chưa
    else fixed=rồi; fi
    cyc=$(journalctl -b --no-pager 2>/dev/null | grep -c "deleted to break ordering cycle")
    boot=$(date -d "$(who -b | awk "{print \$3\" \"\$4}")" +%s 2>/dev/null)
    # Phải đọc lần systemd BẮT ĐẦU chạy unit trong journal của lần boot này.
    # `ActiveEnterTimestamp` là lần kích hoạt GẦN NHẤT, nên máy nào bị restart
    # tay sau đó sẽ báo trễ hàng trăm phút dù lúc boot nó lên bình thường.
    sts=$(journalctl -b -u ocr-all --no-pager -o short-unix 2>/dev/null \
          | grep -m1 "Starting" | cut -d" " -f1 | cut -d. -f1)
    if [ -n "$sts" ] && [ -n "$boot" ]; then lag=$(( (sts - boot) / 60 )); else lag=-1; fi
    echo "$has_jtop|$fixed|$cyc|$lag|$(systemctl is-active ocr-all 2>/dev/null)|$(systemctl is-active ocr-backend 2>/dev/null)"
  ' 2>/dev/null
}

row() {  # tên | có jtop | đã sửa | vòng bị cắt | trễ | ocr-all | backend
  IFS='|' read -r j f c lag a b <<<"$2"
  if [ -z "$j" ]; then printf "  %-11s KHÔNG KẾT NỐI ĐƯỢC\n" "$1"; return; fi
  case "$lag" in
    ''|*[!0-9-]*) t="không đọc được" ;;
    -*)           t="chưa chạy lần nào" ;;
    *) if [ "$lag" -le 5 ]; then t="tự lên sau boot ${lag}p"
       else                      t="KHÔNG tự lên (trễ ${lag}p = gõ tay)"; fi ;;
  esac
  printf "  %-11s jtop:%-6s đã sửa:%-5s vòng:%-2s | %-34s | ocr-all:%s backend:%s\n" \
    "$1" "$j" "$f" "$c" "$t" "$a" "$b"
}

# macOS vẫn là bash 3.2, không có mảng liên kết. Ghi ra file tạm cho gọn.
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "════ TRƯỚC ════"
for h in "${HOSTS[@]}"; do
  probe "$h" > "$TMP/$h"
  row "$h" "$(cat "$TMP/$h")"
done

if [ "$MODE" = check ]; then
  echo
  echo "Chỉ xem. Bỏ --check để sửa thật."
  exit 0
fi

if [ "$MODE" = print ]; then
  echo
  echo "Dán từng dòng vào terminal, mỗi máy sẽ hỏi mật khẩu sudo:"
  echo
  for h in "${HOSTS[@]}"; do
    [ -s "$TMP/$h" ] || { echo "  # $h: không kết nối được"; continue; }
    echo "  scp $AGENT $h:$REMOTE && ssh -t $h 'sudo bash $REMOTE'"
  done
  exit 0
fi

# sudo cần TTY để hỏi mật khẩu. Tiền tố `!` của Claude Code và mọi kiểu chạy
# nền đều không cấp TTY, và khi đó ssh -t bỏ cuộc với "Pseudo-terminal will not
# be allocated". Chặn ngay ở đây thay vì để hỏng lần lượt ở từng máy.
if [ ! -t 0 ]; then
  cat <<'NOTTY'

DỪNG: không có terminal, sudo sẽ không hỏi được mật khẩu.

Mở Terminal.app hoặc iTerm rồi chạy:

    cd ~/Source/Projects/ocr_datecode
    bash scripts/fix_boot_cycle_fleet.sh

Hoặc lấy sẵn lệnh cho từng máy để tự dán:

    bash scripts/fix_boot_cycle_fleet.sh --print
NOTTY
  exit 3
fi

if [ ! -t 0 ]; then
  echo
  echo "DỪNG: không có terminal nên sudo không hỏi được mật khẩu."
  echo "Mở Terminal.app rồi chạy lại. Tiền tố \`!\` của Claude Code không cấp TTY."
  exit 3
fi

echo
echo "════ SỬA ════"
echo "Mỗi máy sẽ hỏi mật khẩu sudo một lần."
echo
for h in "${HOSTS[@]}"; do
  echo "──────── $h ────────"
  if [ ! -s "$TMP/$h" ]; then
    echo "  bỏ qua, không kết nối được"
    continue
  fi
  if ! scp -q -o ConnectTimeout=10 "$AGENT" "$h:$REMOTE"; then
    echo "  không copy được script sang máy, bỏ qua"
    continue
  fi
  arg=""; [ "$MODE" = undo ] && arg="--undo"
  # -t để sudo hỏi mật khẩu ngay trên terminal của bạn.
  "${SSH[@]}" -t "$h" "sudo bash $REMOTE $arg" || echo "  máy này chưa xong, xem thông báo ở trên"
  if [ "$START" -eq 1 ] && [ "$MODE" = fix ]; then
    st=$("${SSH[@]}" -o BatchMode=yes "$h" 'systemctl is-active ocr-all' 2>/dev/null)
    if [ "$st" != "active" ]; then
      echo "  ocr-all đang $st, bật lên (sẽ mở Firefox trên màn hình máy đó)"
      "${SSH[@]}" -t "$h" 'sudo systemctl start ocr-all'
    fi
  fi
  echo
done

echo "════ SAU ════"
for h in "${HOSTS[@]}"; do row "$h" "$(probe "$h")"; done

cat <<'NOTE'

════ BƯỚC XÁC MINH ════

Cột "vòng" và "đã sửa" ở trên mới chỉ nói đồ thị phụ thuộc đã hợp lệ. Lỗi này
chỉ lộ ra lúc khởi động máy, nên phải reboot thật một máy mới kết luận được:

    ssh -t LineTine 'sudo reboot'
    # đợi máy lên, rồi:
    bash scripts/fix_boot_cycle_fleet.sh --check LineTine

Cột giữa phải hiện "tự lên sau boot Np" với N nhỏ. Tin-2 là mốc đối chứng:
trước khi sửa nó đã ra "tự lên sau boot 2p" vì nó là máy duy nhất không dính
vòng.

Nếu vẫn "KHÔNG tự lên" thì gỡ bản sửa rồi báo lại:

    bash scripts/fix_boot_cycle_fleet.sh --undo
NOTE


# làm 1, 2 (nên trả ra thông báo popup là tiếng việt, thông báo cách giải quyết vấn đề nếu ko lấy được frame, 1 là disconnect và connect camera bên tab
#   camera, 2 là load lại recipe để chụp hình, 3 khẩn cấp (restart ocr-all) hiện nút lên để người dùng bấm sau đó counter đếm ngược để restart)
# cho hiện 1 nút khẩn cấp phía phải trên cùng màng hình, khi bấm vào nó sẽ restart lại service 