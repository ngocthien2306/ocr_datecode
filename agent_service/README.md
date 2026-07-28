# AI Agent Service

FastAPI riêng cho hệ AI Agent, tách khỏi backend chính để phát triển / restart
agent không đụng tới dây chuyền đang chạy.

| | Backend | Agent Service |
|---|---|---|
| Port | 8000 | **8100** |
| Package | `app` | `agent_app` |
| Chạy | `backend/run.sh` | `agent_service/run.sh` |

Bản agent cũ trong `backend/app/agent/` **vẫn còn nguyên và vẫn chạy** — hai bên
song song cho tới khi bạn quyết định gỡ (xem [Chuyển FE sang](#chuyển-fe-sang)).

## Chạy

```bash
cd agent_service
cp .env.sample .env          # rồi điền SECRET_KEY + OPENAI_API_KEY
pip3 install -r requirements.txt
./run.sh                     # http://localhost:8100/docs
```

`.env` phải trùng backend ở 4 giá trị: `MONGODB_URL`, `DATABASE_NAME`,
`SECRET_KEY`, `ALGORITHM`. Sai `SECRET_KEY` ⇒ mọi request trả 401 vì không
verify được JWT do backend phát hành.

## UI test nhanh

```
http://localhost:8100/test              (nội bộ)
https://audiobook.a.pinggy.link/test    (tunnel, đưa khách hàng test)
```

Trang HTML đơn (`static/test.html`), không phụ thuộc gì bên ngoài. Serve từ
chính service để cùng origin — mở `file://` trực tiếp sẽ vướng CORS.

Ô "Agent service" **tự điền bằng `location.origin`**, nên cùng một file chạy
được ở cả hai URL trên mà không phải sửa. Đừng hardcode `http://localhost:8100`
vào đó: qua tunnel trang chạy HTTPS, gọi sang `http://` sẽ bị trình duyệt chặn
vì mixed content.

Ô tài khoản để trống có chủ đích — trang này giờ hướng ra khách hàng, prefill
`admin`/`admin123` là phát cho họ luôn tài khoản quản trị.

Có sẵn: đăng nhập lấy token từ backend, chọn agent, quản lý session, 8 câu hỏi
mẫu bấm-một-phát, hiển thị `options` / `suggestions` thành nút bấm, badge
`tool_calls` để soi agent gọi tool nào với tham số gì, và JSON thô của lượt gần
nhất. Dùng để nghiệm thu trước khi tích hợp vào `frontend-ts`.

## Ranh giới với backend

Service này **không import gì từ package `app` của backend**. Ba điểm chạm duy nhất:

1. **MongoDB (dùng chung DB)** — đọc `inference_results`, `recipe_loads`,
   `users`; sở hữu riêng `agent_conversations`.
2. **JWT** — service **tự cấp token** qua `POST /api/auth/login` (cùng đường
   dẫn, cùng form payload, cùng response shape với backend). Xác thực mật khẩu
   bcrypt thẳng trên collection `users`, ký bằng `SECRET_KEY` chung. Không gọi
   backend, nên đăng nhập được kể cả khi backend đang restart.

   ⚠️ Token cấp ở đây **dùng được luôn trên backend production** (đã kiểm
   chứng: `:8000/api/users/me` → 200). Tức `:8100` là cửa xác thực thứ hai vào
   hệ thống. Đổi `SECRET_KEY` thì phải đổi cả hai nơi.

   Hệ quả khi mở tunnel công khai (`audiobook.a.pinggy.link`): ai đăng nhập
   được ở đó thì token cũng dùng được trên backend. Nên đưa khách hàng một tài
   khoản `operator` riêng thay vì `admin` — endpoint của agent chỉ cần
   "đã đăng nhập", còn `RoleChecker` bên backend sẽ chặn `operator` khỏi các
   thao tác quản trị. Đóng tunnel khi hết phiên demo.
3. **HTTP → backend, đúng một việc**: đọc trạng thái WebSocket của camera
   service. `camera_ws_manager` là singleton in-memory chỉ tồn tại trong
   process backend, không thể thấy từ process khác.

Vì (3), backend cần endpoint read-only sau (**đã thêm sẵn, chờ restart backend
mới có hiệu lực**):

```python
# backend/app/api/endpoints/system.py
@router.get("/system/camera-ws-status", tags=["System"])
async def camera_ws_status():
    from app.api.websocket.camera_ws import camera_ws_manager
    return {"connected": camera_ws_manager.is_connected()}
```

Khi backend chưa restart, `check_service_status` trả `status: "unknown"` +
`websocket_connected: null` — cố ý phân biệt với `"degraded"` (chạy nhưng chưa
kết nối). Không bịa ra `false`.

## API

Prefix giữ y hệt backend cũ (`/api/agent/...`) nên FE chỉ cần đổi base URL.

| Method | Path | Auth | Ghi chú |
|---|---|---|---|
| POST | `/api/auth/login` | ❌ | form `username`/`password` → JWT. Không cần backend |
| GET | `/api/auth/me` | ✅ | user của token hiện tại |
| POST | `/api/agent/chat` | ✅ | agent_id: `orchestrator` \| `service_management` \| `historical_analytics` |
| GET | `/api/agent/agents` | ✅ | |
| GET | `/api/agent/health` | ❌ | thêm `backend_reachable` |
| GET | `/api/agent/service/status` | ✅ | gọi thẳng tool, **không tốn token LLM** |
| GET | `/api/agent/conversations` | ✅ | |
| DELETE | `/api/agent/conversation/{session_id}` | ✅ | |
| POST | `/api/agent/chat/stream` | ✅ | experimental, **không lưu history** |
| GET | `/health` | ❌ | cho systemd / monitoring |

## Tools

`historical_analytics` (5):

| Tool | Dùng cho |
|---|---|
| `list_recipes` | liệt kê recipe đang có sản lượng — khi user chưa nêu recipe nào |
| `get_pass_fail_stats` | pass/fail + xu hướng theo giờ/ngày/tuần/tháng, lọc theo recipe |
| `get_production_summary` | tổng quan gom theo recipe / **camera** / giờ; lọc theo **khung giờ** và recipe |
| `get_recipe_load_history` | ai load/stop recipe, chạy bao lâu |
| `explain_failures` | **vì sao fail** — gom theo bước kiểm tra trượt, kèm cặp expected→recognized |

`service_management` (4): `check_service_status`, `get_service_logs`,
`start_service`, `stop_service` (hai cái sau chưa test — xem [Việc còn nợ](#việc-còn-nợ)).

`explain_failures` mổ 4 bước kiểm tra của mỗi frame — `text_verification`
(OCR đọc sai chuỗi), `char_verification` (ký tự dưới ngưỡng),
`template_verification` (similarity < threshold), `product_verification`
(không nhận ra sản phẩm/nhãn) — nên trả lời được "tại sao" chứ không chỉ đếm.
Ví dụ thật: trong 121 sản phẩm fail của một khung 2 tiếng, 103 do OCR đọc sai,
với cặp lệch nhiều nhất là `06203-11704-V` → `06208-11704-V` (nhầm 3 thành 8).

## Images & Charts — trực quan hoá

`POST /api/agent/chat` trả thêm `images` và `charts`.

```jsonc
{
  "images": [
    { "url": "/api/uploads/inference_results/<recipe>/<ngày>/<camera>/fail_..._viz.jpg",
      "caption": "Camera 24026290 · 17:23:30 · mong '06203-11704-V' → đọc '06208-11704-V'",
      "recipe": "minced onion (Copy 2)" }
  ],
  "charts": [
    { "type": "bar", "unit": "sp",
      "title": "Sản phẩm FAIL theo camera · 2026-07-22 16:00–18:00",
      "series": [ { "label": "24026290", "value": 121, "sub": "95.82% pass · 2,896 sp" } ] }
  ]
}
```

**Cả hai đều suy TẤT ĐỊNH từ kết quả tool, không qua LLM** (`core/attachments.py`).
Để LLM tự bịa số cho biểu đồ thì có nguy cơ hình một đằng số một nẻo — mà biểu đồ
là thứ người ta tin ngay bằng mắt, không kiểm lại.

- `images` ← `explain_failures` trả kèm `samples` (tối đa 8 frame fail gần nhất)
- `charts` ← `get_production_summary` (fail theo camera/recipe/giờ),
  `get_pass_fail_stats` (pass rate theo thời gian, cần ≥2 mốc),
  `explain_failures` (nguyên nhân + frame fail theo camera)

`chart` cố tình chỉ có MỘT dạng — danh sách nhãn/giá trị — để FE chỉ viết một bộ
render. So sánh hai kỳ thì LLM gọi tool hai lần → hai chart cạnh nhau, không cần
kiểu chart riêng. Tiêu đề **bắt buộc kèm nhãn kỳ** (`· 2026-07-21`,
`· 2026-07-22 16:00–18:00`); thiếu nó thì hai biểu đồ so sánh trông giống hệt
nhau và mất sạch ý nghĩa.

Ảnh serve tại `/api/uploads/...` — agent service mount **cùng thư mục**
`backend/uploads` mà backend dùng. Bắt buộc phải mount: tunnel chỉ mở `:8100`,
không có nó thì khách hàng xem qua tunnel sẽ thấy ảnh vỡ hết. URL để **tương
đối** nên tự resolve theo origin, chạy đúng ở cả localhost lẫn tunnel HTTPS.

## Options — nút chọn khi câu hỏi chưa rõ

`POST /api/agent/chat` có thể trả thêm `options`. FE render thành nút bấm;
**bấm nút = gửi `value` vào đúng endpoint đó như một tin nhắn thường**, cùng
`session_id`. Không cần xử lý gì đặc biệt, và user vẫn gõ tay được như cũ.

```jsonc
{
  "response": "Có 3 recipe tên gần giống 'minced onion'. Bạn muốn xem recipe nào?",
  "options": [
    { "label": "minced onion (Copy 2)", "hint": "30.562 sản phẩm",
      "value": "Recipe 6a60026f3d0ad35f61716cea (minced onion (Copy 2))" },
    { "label": "minced onion",          "hint": "19.529 sản phẩm", "value": "..." }
  ],
  "session_id": "...", "tool_calls": [...], "timestamp": "..."
}
```

`value` nhúng sẵn ObjectId để lượt sau khớp đúng một recipe — `_id_or_name()`
thấy chuỗi 24 ký tự hex sẽ lọc theo ID thay vì khớp tên mờ.

Hai lúc `options` xuất hiện:

1. **Tên khớp nhiều recipe.** `'minced onion'` hiện trúng 3–5 recipe riêng biệt
   (`minced onion`, `minced onion (Copy)`, `(Copy 2)`, `Minced Onion`,
   `Minced Onion 1`). Trước đây tool cộng gộp hết rồi trả một con số duy nhất —
   user không hề biết mình đang xem số của nhiều recipe trộn lại. Giờ tool trả
   `needs_disambiguation` và bắt hỏi lại.
2. **User chưa nêu recipe nào** mà lại hỏi thống kê theo recipe → agent gọi
   `list_recipes()` rồi đưa danh sách.

Hỏi tổng toàn dây chuyền ("hôm nay sản xuất bao nhiêu") thì **không** hỏi lại —
trả lời thẳng, `options` là `null`.

Luồng dữ liệu: tool trả `needs_disambiguation`/`recipes` → `execute_tools` gọi
`core/ui_options.py` → đặt vào `state.context["ui_options"]` → `chat.py` đọc ra.
Orchestrator phải chuyển tiếp `context` khi route, nếu chỉ trả `messages` là
mất nút.

## Suggestions — chip gợi ý câu hỏi tiếp theo

Đi kèm `options`, nhưng khác ý nghĩa:

| | ý nghĩa | khi nào |
|---|---|---|
| `options` | **bắt buộc chọn** một, không chọn thì bí | tên recipe mơ hồ, chưa nêu recipe |
| `suggestions` | tuỳ chọn, thích thì bấm không thì gõ | sau mọi câu trả lời |

```jsonc
{
  "response": "Hôm nay 51.370 sản phẩm, 78.32% pass…",
  "suggestions": ["So sánh với hôm qua", "Xu hướng 7 ngày qua", "Camera nào fail nhiều nhất?"]
}
```

Chip cũng là chuỗi gửi thẳng vào `/api/agent/chat` như tin nhắn thường.
Khi có `options` thì `suggestions` = `null` (bắt chọn xong đã, đừng gây phân tán).

Cách lấy, hai tầng:
1. LLM tự sinh, bọc trong `[SUGGESTIONS]…[/SUGGESTIONS]`. Backend parse rồi **gỡ
   khỏi text** trước khi trả FE và trước khi lưu DB — user không bao giờ thấy thẻ.
   Dùng khối phân cách thay vì parse "1. 2. 3." trong văn xuôi vì danh sách đánh
   số xuất hiện đầy trong nội dung thật (bước khắc phục, dòng log, bảng số liệu).
2. LLM quên khối → suy ra từ tool vừa chạy (`core/suggestions.py::_FALLBACK`).
   Tất định, luôn hợp lệ.

**Lọc an toàn**: gợi ý là chip bấm-một-phát, không có bước xác nhận. Server chặn
mọi gợi ý chứa động từ phá huỷ (dừng/stop/restart/tắt/xoá/reset) — LLM từng tự
đề xuất *"Dừng Camera service"*, bấm vào là dừng dây chuyền đang chạy. Prompt
cũng dặn, nhưng lọc ở server mới là chốt chặn thật.

## Khác biệt so với bản trong backend

Port sang đây có sửa 4 điểm, phần còn lại giữ nguyên hành vi:

1. **`api_key` truyền tường minh** (`core/llm.py`). Bản cũ tạo
   `ChatOpenAI(...)` không truyền key nên dựa vào biến môi trường
   `OPENAI_API_KEY` — mà pydantic-settings chỉ nạp `.env` vào `settings`, không
   export ra `os.environ`. Nghĩa là bản cũ chỉ chạy khi có ai đó `export` tay.
2. **Chặn IDOR trên hội thoại.** Mọi query/xoá `agent_conversations` đều kèm
   `user_id`. Bản cũ chỉ lọc theo `session_id` — mà `session_id` do client gửi
   và format đoán được (`session_{user_id}_{timestamp}`), nên user này đọc/xoá
   được chat của user khác.
3. **Giới hạn history** `MAX_HISTORY_MESSAGES` (mặc định 40). Bản cũ replay
   toàn bộ message mỗi lượt ⇒ context window phình theo tuổi đời session, chi
   phí token tăng tuyến tính.
4. **Index thật** cho `agent_conversations` (`session_id` unique,
   `user_id+updated_at`). `Field(..., index=True)` của Pydantic là no-op, không
   tạo index gì cả.

## Chuyển FE sang

`frontend-ts/src/services/agentService.ts` đang trỏ vào backend :8000. Đổi base
URL của riêng các call `/api/agent/*` sang `http://<host>:8100`.

⚠️ **KHÔNG xoá được `backend/app/agent/`.** Watchdog của backend import ngược
vào đó làm cơ chế khôi phục:

```
backend/app/services/camera_service_supervisor.py:144
    from app.agent.tools.service_tools import stop_service, start_service
```

Gỡ đi là watchdog mất đường kill-respawn khi `systemctl` không dùng được. Muốn
dọn thì phải chuyển 2 hàm đó ra một module hạ tầng của backend trước
(`app/services/`), rồi mới xoá phần agent còn lại — `app/api/endpoints/agent.py`
và dòng `include_router(agent...)` trong `app/main.py` thì gỡ được ngay.

## Việc còn nợ

Các vấn đề kiến trúc dưới đây được port nguyên trạng, **chưa sửa** — giờ sửa
được trong source này mà không đụng production:

- `requires_approval=True` trên `start_service`/`stop_service` khai báo nhưng
  không chỗ nào enforce; "xác nhận" hiện chỉ nằm trong system prompt.
- `AgentState.messages` thiếu reducer `add_messages` ⇒ `execute_tools` bị viết
  lại 3 lần theo 3 kiểu khác nhau; `historical_agent` gọi `tool.func(**args)`
  nên bỏ qua validation của `args_schema`. LangGraph có sẵn `ToolNode`.
- Orchestrator route bằng parse JSON free-text thay vì structured output, và
  hardcode danh sách agent thay vì đọc từ `AgentRegistry`.
- Node function đều là sync ⇒ chạy trong threadpool; orchestrator còn gọi
  `graph.invoke()` lồng bên trong (sync trong sync).
