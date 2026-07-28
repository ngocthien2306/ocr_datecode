"""
Agent Service configuration.

Standalone: does NOT import backend's Settings. The only values that MUST be
kept in sync with backend/.env are SECRET_KEY + ALGORITHM (JWT được backend
phát hành, service này chỉ verify) và MONGODB_URL/DATABASE_NAME (dùng chung DB).
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- Shared with backend (PHẢI trùng giá trị trong backend/.env) --------
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 360
    # Múi giờ user nghĩ khi nói "hôm nay". DB lưu timestamp naive-UTC nên mọi
    # mốc ngày phải quy đổi trước khi query — xem tools/analytics_tools.py
    TIMEZONE: str = "Asia/Ho_Chi_Minh"

    # ---- Agent service riêng ------------------------------------------------
    APP_NAME: str = "OCR Datecode Agent Service"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8100

    # Backend đang chạy — dùng để đọc state in-process của nó (WebSocket status)
    BACKEND_URL: str = "http://localhost:8000"
    BACKEND_TIMEOUT: float = 3.0

    OPENAI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gpt-4o-mini"

    # Số message lịch sử tối đa nạp lại vào LLM mỗi lượt chat.
    # Chặn context window phình vô hạn theo tuổi đời session.
    MAX_HISTORY_MESSAGES: int = 40

    CORS_ORIGINS: List[str] = ["*"]

    # Đường dẫn tuyệt đối tới repo root (chứa ai_services/, backend/, logs/).
    # Để trống = tự suy ra từ vị trí file này.
    PROJECT_ROOT: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def project_root(self) -> Path:
        if self.PROJECT_ROOT:
            return Path(self.PROJECT_ROOT).resolve()
        # agent_service/agent_app/core/config.py → core → agent_app → agent_service → repo root
        return Path(__file__).resolve().parents[3]


settings = Settings()
