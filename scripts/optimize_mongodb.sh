#!/bin/bash
# ============================================================================
# optimize_mongodb.sh — Tune MongoDB (Docker) cho Jetson / máy RAM thấp.
#
# Làm 3 việc, IDEMPOTENT (chạy lại nhiều lần vô hại):
#   1) Cap WiredTiger cache NGAY (live, không cần restart container).
#   2) Ghi cache cap vào docker-compose.yml để GIỮ sau reboot.
#   3) Tạo index + TTL cho inference_results / action_logs (tối ưu query + tự
#      dọn data cũ). (BE cũng tự tạo lúc startup; tạo ở đây để DB tối ưu ngay
#      cả khi máy chưa deploy code mới.)
#
# Cách dùng:
#   ./optimize_mongodb.sh
#   MONGO_CACHE_GB=1 INFERENCE_TTL_DAYS=45 ./optimize_mongodb.sh   # override
#
# KHÔNG dùng `set -e`: tự xử lý lỗi từng bước để 1 bước hỏng không phá cả script.
# ============================================================================
set -uo pipefail

# ---- Config (override qua biến môi trường) --------------------------------
CONTAINER="${MONGO_CONTAINER:-mongodb}"
MUSER="${MONGO_USER:-admin}"
MPASS="${MONGO_PASS:-password}"          # mật khẩu MongoDB DB (KHÔNG phải mongo-express)
MDB="${MONGO_DB:-ocr_datecode_db}"
CACHE_GB="${MONGO_CACHE_GB:-1.5}"
INFERENCE_TTL_DAYS="${INFERENCE_TTL_DAYS:-30}"
ACTIONLOG_TTL_DAYS="${ACTIONLOG_TTL_DAYS:-90}"

CACHE_MB=$(awk "BEGIN{printf \"%d\", $CACHE_GB*1024}")

# mongosh helpers ------------------------------------------------------------
run_admin() {  # JS chạy trên admin (serverStatus / setParameter)
    docker exec "$CONTAINER" mongosh -u "$MUSER" -p "$MPASS" \
        --authenticationDatabase admin --quiet --eval "$1"
}
run_db() {     # JS chạy trên DB ứng dụng
    docker exec "$CONTAINER" mongosh -u "$MUSER" -p "$MPASS" \
        --authenticationDatabase admin --quiet "$MDB" --eval "$1"
}

echo "=============================================="
echo " Optimize MongoDB: container=$CONTAINER db=$MDB"
echo " cache=${CACHE_GB}GB  TTL inference=${INFERENCE_TTL_DAYS}d  action_logs=${ACTIONLOG_TTL_DAYS}d"
echo "=============================================="

# 0) Container có chạy không -------------------------------------------------
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "❌ Container '$CONTAINER' không chạy. Sửa MONGO_CONTAINER rồi thử lại."
    exit 1
fi
# Thử kết nối + auth
if ! run_admin 'db.runCommand({ping:1}).ok' >/dev/null 2>&1; then
    echo "❌ Không kết nối/auth được MongoDB. Kiểm tra MONGO_USER/MONGO_PASS."
    exit 1
fi
echo "✅ Kết nối MongoDB OK"

# 1) Cap cache NGAY (live) ---------------------------------------------------
echo "--- [1/3] Set WiredTiger cache = ${CACHE_GB}GB (live) ---"
run_admin "db.adminCommand({setParameter:1, wiredTigerEngineRuntimeConfig:'cache_size=${CACHE_MB}M'})" \
    && echo "✅ Đã set cache live" \
    || echo "⚠️  Set cache live thất bại (xem log trên)"
run_admin 'print("   cache max = "+(db.serverStatus().wiredTiger.cache["maximum bytes configured"]/1048576).toFixed(0)+"MB | dang dung = "+(db.serverStatus().wiredTiger.cache["bytes currently in the cache"]/1048576).toFixed(0)+"MB")'

# 2) Ghi cache cap vào docker-compose (giữ sau reboot) -----------------------
echo "--- [2/3] Persist cache vào docker-compose ---"
COMPOSE=$(docker inspect "$CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}' 2>/dev/null)
if [ -z "$COMPOSE" ] || [ ! -f "$COMPOSE" ]; then
    echo "⚠️  Không tìm thấy compose file (container không do compose quản lý?). Bỏ qua persist."
    echo "    Cache vẫn đang ở ${CACHE_GB}GB (live) nhưng SẼ MẤT khi container restart."
elif grep -q "wiredTigerCacheSizeGB" "$COMPOSE"; then
    echo "✅ Compose đã có cache flag từ trước ($COMPOSE) — không sửa."
else
    cp "$COMPOSE" "$COMPOSE.bak.$(date +%s)"
    # Chèn dòng command ngay dưới 'image: mongo:' (chỉ khớp service mongodb,
    # KHÔNG khớp mongo-express). Giữ --auth --bind_ip_all như entrypoint mặc định.
    sed -i "/image: mongo:/a\\    command: [\"mongod\", \"--auth\", \"--bind_ip_all\", \"--wiredTigerCacheSizeGB\", \"$CACHE_GB\"]" "$COMPOSE"
    if grep -q "wiredTigerCacheSizeGB" "$COMPOSE"; then
        echo "✅ Đã ghi cache cap vào $COMPOSE (backup .bak.*)"
        echo "   ⚠️  Để áp dụng vĩnh viễn cần recreate (CÓ downtime vài giây):"
        echo "        docker compose -f \"$COMPOSE\" up -d $CONTAINER"
    else
        echo "⚠️  Chèn command thất bại (cấu trúc compose khác?). Kiểm tra thủ công $COMPOSE."
    fi
fi

# 3) Index + TTL (idempotent) -----------------------------------------------
echo "--- [3/3] Tạo index + TTL ---"
run_db "
var INF_SEC = ${INFERENCE_TTL_DAYS}*86400;
var LOG_SEC = ${ACTIONLOG_TTL_DAYS}*86400;

// Tạo TTL an toàn: nếu đã có index single-field (không TTL) trên field -> drop & tạo lại.
function ensureTTL(coll, field, sec){
  var key = {}; key[field] = -1;
  try { db[coll].createIndex(key, {expireAfterSeconds: sec}); print('  TTL '+coll+'.'+field+' OK ('+(sec/86400)+'d)'); return; }
  catch(e){
    if (e.code !== 85 && e.code !== 86 && e.codeName !== 'IndexOptionsConflict' && e.codeName !== 'IndexKeySpecsConflict'){
      print('  ! TTL '+coll+'.'+field+' loi: '+e.message); return;
    }
  }
  db[coll].getIndexes().forEach(function(ix){
    var ks = Object.keys(ix.key);
    if (ks.length === 1 && ks[0] === field) { db[coll].dropIndex(ix.name); }
  });
  db[coll].createIndex(key, {expireAfterSeconds: sec});
  print('  TTL '+coll+'.'+field+' recreated ('+(sec/86400)+'d)');
}

// --- inference_results ---
db.inference_results.createIndex({timestamp:-1});
db.inference_results.createIndex({recipe_id:1});
db.inference_results.createIndex({product_pass_fail:1});
// Covering index cho summary/timeseries -> aggregation doc tu index, KHONG fetch document.
db.inference_results.createIndex({created_at:1, product_pass_fail:1, recipe_id:1, recipe_name:1});
print('  inference_results: base + covering index OK');
ensureTTL('inference_results','created_at', INF_SEC);

// --- action_logs ---
db.action_logs.createIndex({user_id:1});
db.action_logs.createIndex({action_type:1});
db.action_logs.createIndex({resource_type:1});
ensureTTL('action_logs','timestamp', LOG_SEC);

// Preview: TTL se don bao nhieu
var d = new Date(Date.now() - INF_SEC*1000);
print('  inference_results cu hon '+${INFERENCE_TTL_DAYS}+' ngay (TTL se xoa dan): '
      + db.inference_results.countDocuments({created_at:{\$lt:d}}) + ' / '
      + db.inference_results.estimatedDocumentCount());
"

echo "--- Verify: index hien tai tren inference_results ---"
run_db 'db.inference_results.getIndexes().forEach(function(i){print("  "+i.name+(i.expireAfterSeconds!=null?"  [TTL "+(i.expireAfterSeconds/86400)+"d]":""))})'

echo ""
echo "✅ XONG. Cache đã cap live; TTL monitor sẽ tự dọn data cũ (~mỗi 60s)."
echo "   Nhớ: muốn cache giữ sau reboot thì recreate container theo lệnh ở bước [2/3]."
