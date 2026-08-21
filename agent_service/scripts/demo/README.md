# Script dữ liệu demo

Tạo user test, hồ sơ nhân sự, ảnh chân dung và hoạt động theo ca — dùng để test
các tool đọc audit log của agent và phần thẻ nhân sự trên `/test`.

**Hướng dẫn đầy đủ, kèm thứ tự chạy và lý do từng bước:
[`../../docs/DEMO_DATA.md`](../../docs/DEMO_DATA.md)**

⚠️ Các script này **cố ý không chạm** tới `load_recipe` / `stop_recipe` / recipe /
camera. Dây chuyền đang chạy thật. Thêm script mới thì giữ nguyên ràng buộc đó.

Kết quả trung gian ghi vào `_out/` (đã gitignore — chứa mật khẩu dạng rõ).
