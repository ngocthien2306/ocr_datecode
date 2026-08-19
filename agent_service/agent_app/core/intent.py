"""
Bảng cụm từ chỉ rõ ý định → agent phụ trách. MỘT nguồn sự thật duy nhất.

Dùng cho hai việc khác nhau, và đó là lý do phải tách ra file riêng:

1. `orchestrator_agent` — đường tắt: câu hỏi khớp một cụm rõ ràng thì gọi thẳng
   agent đó, BỎ QUA lượt LLM chọn tool. Tiết kiệm ~1,2s mỗi câu.
2. `core/reroute` — phát hiện route sai: ý định trong câu hỏi lệch với agent đã trả
   lời thì bày nút hỏi lại.

Hai chỗ này từng có hai bảng riêng. Để vậy thì chúng sẽ lệch nhau, và lệch theo cách
tệ nhất: đường tắt gửi câu hỏi tới agent A trong khi reroute lại khẳng định câu đó
thuộc agent B, nên người dùng nhận câu trả lời kèm luôn một nút nói rằng câu trả lời
này sai chỗ.

## Nguyên tắc chọn cụm

Chỉ nhận cụm gần như KHÔNG THỂ hiểu theo nghĩa khác. Bảng càng rộng thì đường tắt
càng dễ gửi sai chỗ, mà đường tắt không có LLM để tự sửa. Thà bỏ lọt (rơi về lượt
LLM, chậm hơn 1,2s) hơn là đoán sai.

Cụ thể: không nhận cụm một từ như "lỗi", "log", "ca", "camera" — đó đúng là những từ
đã gây định tuyến sai. Chỉ nhận cụm nhiều từ đã cố định nghĩa.
"""

from typing import Optional, Tuple

# (agent_id, các cụm từ). Thứ tự trong danh sách là thứ tự xét, nên cụm hẹp và chắc
# chắn nhất đặt trước.
INTENT_PHRASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # Người dùng / audit. Đặt TRƯỚC nhóm sản xuất vì "bao nhiêu người đăng nhập"
    # nghe như câu thống kê và từng bị route sang agent sản xuất — agent đó không có
    # tool nào về đăng nhập nên trả lời "không có dữ liệu", một câu SAI.
    ("log_analysis", (
        "bao nhiêu người đăng nhập", "ai đăng nhập", "ai đã đăng nhập",
        "ai load recipe", "ai sửa recipe", "ai xoá recipe", "ai tạo user",
        "lịch sử thao tác", "audit log", "nhật ký thao tác",
        "dung lượng log", "log nặng bao nhiêu",
        "how many users logged in", "who logged in", "who loaded a recipe",
        "audit trail", "log size", "disk usage",
    )),
    # Sản phẩm fail — nằm trong database, KHÔNG nằm trong file log.
    ("historical_analytics", (
        "sản phẩm lỗi", "sản phẩm fail", "sản phẩm không đạt", "hàng lỗi",
        "nguyên nhân fail", "vì sao fail", "tại sao fail",
        "pass rate", "sản lượng", "bao nhiêu sản phẩm",
        "bản giao ca", "giao ca", "đạt chỉ tiêu", "còn thiếu bao nhiêu",
        "xuất báo cáo", "tạo báo cáo", "xuất excel", "xuất pdf",
        "failed product", "failed unit", "fail causes", "how many units",
        "shift handover", "export report", "on target",
    )),
    ("equipment_health", (
        "xung reject", "cơ cấu đẩy", "đẩy phôi", "thời gian đẩy",
        "trigger có ổn", "bị bỏ sót", "mất ảnh", "nhịp dây chuyền",
        "cảm biến có ổn", "module nào lỗi", "kiểm tra thiết bị",
        # Các cách nói về "thiết bị có vấn đề". Thiếu chúng thì câu HAI NGUỒN kiểu
        # "sản lượng có bị ảnh hưởng bởi lỗi thiết bị không" chỉ khớp nhóm sản xuất,
        # bị đường tắt bắt mất, và mất luôn khả năng gọi hai agent.
        "lỗi thiết bị", "thiết bị lỗi", "sự cố thiết bị", "thiết bị có vấn đề",
        "máy móc có vấn đề", "equipment error", "equipment issue",
        "reject pulse", "trigger health", "sensor pulse", "module in error",
    )),
    ("service_management", (
        "camera service có đang chạy", "service có đang chạy",
        "khởi động lại service", "dừng service", "bật service",
        "kết nối websocket",
        "is the camera service running", "restart the service",
    )),
)


def match(text: str) -> Optional[str]:
    """
    Agent mà câu hỏi khớp RÕ RÀNG, hoặc None nếu không chắc.

    Trả về None là kết quả bình thường, không phải thất bại: khi đó tầng trên dùng
    LLM để quyết định. Hàm này chỉ tồn tại để bắt các câu hỏi lặp lại nhiều nhất.
    """
    if not text:
        return None
    low = text.lower()
    hits = {agent for agent, phrases in INTENT_PHRASES
            if any(p in low for p in phrases)}
    # Khớp nhiều agent nghĩa là câu hỏi liên quan nhiều nguồn — đúng lúc CẦN LLM để
    # điều phối, nên trả None. Đường tắt chỉ dành cho câu một nguồn.
    return hits.pop() if len(hits) == 1 else None
