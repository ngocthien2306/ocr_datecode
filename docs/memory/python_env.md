---
name: Python venv path
description: Đường dẫn virtualenv Python user dùng cho project ocr_datecode (training scripts, etc.)
type: project
originSessionId: c61dd732-057b-4335-b145-009b6951e706
---
User dùng venv tại `/Users/ngocthien.ai/envs/event/`. Python interpreter: `/Users/ngocthien.ai/envs/event/bin/python`.

**Why:** User nhiều lần gửi `source /Users/ngocthien.ai/envs/event/bin/activate` để báo hiệu cần dùng venv này. Bash tool không giữ shell state giữa các lệnh nên `source` đơn lẻ vô tác dụng.

**How to apply:** Khi cần chạy Python (training scripts, eval, viz, pip install...), dùng một trong:
- `source /Users/ngocthien.ai/envs/event/bin/activate && python <script>` (gộp 1 lệnh)
- `/Users/ngocthien.ai/envs/event/bin/python <script>` (gọi trực tiếp interpreter)
- `/Users/ngocthien.ai/envs/event/bin/pip install ...`

Không chạy `python` trống vì sẽ lấy system Python sai env.
