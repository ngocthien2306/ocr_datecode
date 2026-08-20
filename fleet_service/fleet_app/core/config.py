"""Cấu hình fleet service."""

import json
from pathlib import Path
from typing import Dict

from pydantic_settings import BaseSettings

REPO = Path(__file__).resolve().parents[3]          # …/ocr_datecode
SERVICE_ROOT = Path(__file__).resolve().parents[2]  # …/fleet_service


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8200

    FLEET_EDGE_USER: str = "admin"
    FLEET_EDGE_PASSWORD: str = ""
    # Chuỗi JSON chứ không phải Dict: pydantic-settings đọc Dict từ .env bằng
    # cú pháp lồng nhau rất dễ sai, còn một chuỗi JSON thì đọc sao ghi vậy.
    FLEET_EDGE_PASSWORD_OVERRIDES: str = "{}"

    EDGE_AGENT_PORT: int = 8100
    EDGE_BACKEND_PORT: int = 8000

    EDGE_TIMEOUT: float = 8.0
    EDGE_CHAT_TIMEOUT: float = 90.0
    # Lần lạnh của rollup phải mổ vài trăm document fail trên Jetson.
    EDGE_ROLLUP_TIMEOUT: float = 30.0

    POLL_INTERVAL: float = 60.0
    STALE_AFTER: float = 180.0

    TAILSCALE_BIN: str = "/usr/local/bin/tailscale"

    class Config:
        env_file = str(SERVICE_ROOT / ".env")
        case_sensitive = True
        extra = "ignore"

    def password_for(self, node_id: str) -> str:
        """Mật khẩu của một máy: override theo node id, không có thì dùng chung."""
        try:
            overrides: Dict[str, str] = json.loads(self.FLEET_EDGE_PASSWORD_OVERRIDES or "{}")
        except json.JSONDecodeError:
            overrides = {}
        return overrides.get(node_id) or self.FLEET_EDGE_PASSWORD


settings = Settings()

DB_PATH = SERVICE_ROOT / "fleet.db"
MACHINES_FILE = SERVICE_ROOT / "config" / "machines.json"
