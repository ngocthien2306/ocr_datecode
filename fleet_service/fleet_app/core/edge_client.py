"""
Client HTTP tới agent service (:8100) và backend (:8000) của từng máy.

Điểm bắt buộc phải đúng, nếu sai thì cả hệ thống chạy mà kết quả vô nghĩa:

**Mỗi máy một JWT riêng.** Agent service TỰ verify token bằng `SECRET_KEY` của
chính nó, không hỏi ai. Trên đội hình hiện tại, 4 Jetson dùng
`change-this-secret-key-in-production` còn PC-Auto-1 dùng
`your-secret-key-here-change-this-in-production`. Không có token dùng chung —
phải đăng nhập riêng từng máy và cache theo node id.

**Không ném exception ra ngoài.** Mọi lời gọi trả về `EdgeResult`. Một máy chết
không được làm hỏng lượt hỏi của bốn máy còn lại; nó phải trở thành một dòng
"máy này không trả lời" trong kết quả.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx

from fleet_app.core.config import settings

logger = logging.getLogger(__name__)

# Token của agent service sống 360 phút (ACCESS_TOKEN_EXPIRE_MINUTES của backend).
# Làm mới sớm hơn nhiều để không bao giờ dùng phải token vừa hết hạn giữa chừng.
_TOKEN_TTL = 60 * 60.0


@dataclass
class EdgeResult:
    """
    Kết quả một lời gọi tới edge.

    `ok=False` luôn kèm `error` đọc được bằng tiếng người — nó sẽ đi thẳng lên
    giao diện, nên "không với tới được từ 10:42" hữu ích hơn `ConnectTimeout`.
    """
    ok: bool
    data: Any = None
    error: Optional[str] = None
    status: Optional[int] = None
    elapsed: float = 0.0


@dataclass
class _Token:
    value: str
    fetched_at: float = field(default_factory=time.monotonic)

    @property
    def fresh(self) -> bool:
        return (time.monotonic() - self.fetched_at) < _TOKEN_TTL


class EdgeClient:
    def __init__(self) -> None:
        self._tokens: Dict[str, _Token] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        # Một client dùng chung để giữ connection pool; mở client mới cho mỗi
        # lời gọi thì bắt tay TCP/TLS lại từ đầu, rất tốn trên link chậm.
        self._client = httpx.AsyncClient(follow_redirects=True)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def forget(self, node_id: str) -> None:
        """Bỏ token đã cache — gọi khi máy trả 401."""
        self._tokens.pop(node_id, None)

    # --- nền tảng ----------------------------------------------------------

    def _url(self, ip: str, port: int, path: str) -> str:
        return f"http://{ip}:{port}{path}"

    async def _request(self, method: str, url: str, *, timeout: float,
                       token: Optional[str] = None, **kw) -> EdgeResult:
        assert self._client is not None, "EdgeClient chưa start()"
        headers = kw.pop("headers", {}) or {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        t0 = time.monotonic()
        try:
            resp = await self._client.request(method, url, timeout=timeout,
                                              headers=headers, **kw)
            elapsed = time.monotonic() - t0
            if resp.status_code >= 400:
                return EdgeResult(False, error=f"HTTP {resp.status_code}",
                                  status=resp.status_code, elapsed=elapsed)
            try:
                return EdgeResult(True, data=resp.json(), status=resp.status_code,
                                  elapsed=elapsed)
            except ValueError:
                return EdgeResult(True, data=resp.text, status=resp.status_code,
                                  elapsed=elapsed)
        except httpx.TimeoutException:
            return EdgeResult(False, error=f"quá {timeout:.0f}s không trả lời",
                              elapsed=time.monotonic() - t0)
        except httpx.HTTPError as e:
            return EdgeResult(False, error=f"không kết nối được ({type(e).__name__})",
                              elapsed=time.monotonic() - t0)

    async def token_for(self, node_id: str, ip: str) -> Optional[str]:
        """
        Token của một máy, tự đăng nhập lại khi hết hạn.

        Khoá theo node id để hai lời gọi song song tới cùng một máy không cùng
        lúc đăng nhập — link tới Jetson chậm, hai lần login thừa là thừa thật.
        """
        cached = self._tokens.get(node_id)
        if cached and cached.fresh:
            return cached.value

        lock = self._locks.setdefault(node_id, asyncio.Lock())
        async with lock:
            cached = self._tokens.get(node_id)
            if cached and cached.fresh:
                return cached.value
            res = await self._request(
                "POST",
                self._url(ip, settings.EDGE_AGENT_PORT, "/api/auth/login"),
                timeout=settings.EDGE_TIMEOUT,
                data={"username": settings.FLEET_EDGE_USER,
                      "password": settings.password_for(node_id)},
            )
            if not res.ok or not isinstance(res.data, dict):
                logger.warning("Đăng nhập thất bại vào %s (%s): %s", node_id, ip, res.error)
                return None
            tok = res.data.get("access_token")
            if not tok:
                return None
            self._tokens[node_id] = _Token(tok)
            return tok

    async def _authed(self, node_id: str, ip: str, port: int, path: str,
                      *, timeout: Optional[float] = None, method: str = "GET",
                      **kw) -> EdgeResult:
        token = await self.token_for(node_id, ip)
        if not token:
            return EdgeResult(False, error="không đăng nhập được (sai mật khẩu?)")
        res = await self._request(method, self._url(ip, port, path),
                                  timeout=timeout or settings.EDGE_TIMEOUT,
                                  token=token, **kw)
        # Token có thể bị vô hiệu khi máy đổi SECRET_KEY hoặc restart backend.
        # Thử lại đúng MỘT lần với token mới, rồi thôi.
        if res.status == 401:
            self.forget(node_id)
            token = await self.token_for(node_id, ip)
            if token:
                res = await self._request(method, self._url(ip, port, path),
                                          timeout=timeout or settings.EDGE_TIMEOUT,
                                          token=token, **kw)
        return res

    # --- các bậc của thang năng lực ----------------------------------------

    async def health(self, ip: str) -> EdgeResult:
        """L1 — không cần token, nên dùng được cả khi mật khẩu sai."""
        return await self._request(
            "GET", self._url(ip, settings.EDGE_AGENT_PORT, "/api/agent/health"),
            timeout=settings.EDGE_TIMEOUT)

    async def agents(self, node_id: str, ip: str) -> EdgeResult:
        """L2 — danh sách agent, suy ra máy trả lời được loại câu hỏi nào."""
        return await self._authed(node_id, ip, settings.EDGE_AGENT_PORT,
                                  "/api/agent/agents")

    async def service_status(self, node_id: str, ip: str) -> EdgeResult:
        return await self._authed(node_id, ip, settings.EDGE_AGENT_PORT,
                                  "/api/agent/service/status")

    async def system_metrics(self, ip: str) -> EdgeResult:
        """Phần cứng. Endpoint của backend, không cần token."""
        return await self._request(
            "GET", self._url(ip, settings.EDGE_BACKEND_PORT,
                             "/api/jetson-monitoring/metrics"),
            timeout=settings.EDGE_TIMEOUT)

    async def rollup(self, node_id: str, ip: str, days: int = 7,
                     causes: bool = True) -> EdgeResult:
        """
        L5 — số liệu sản xuất gọn, KHÔNG qua LLM. Đo được 1,4 KB / 2,0s lạnh,
        0,2s khi edge còn cache.

        Timeout rộng hơn EDGE_TIMEOUT vì lần lạnh phải mổ vài trăm document fail
        trên Jetson; 8s là vừa đủ hụt.
        """
        return await self._authed(node_id, ip, settings.EDGE_AGENT_PORT,
                                  "/api/fleet/rollup",
                                  timeout=settings.EDGE_ROLLUP_TIMEOUT,
                                  params={"days": days,
                                          "causes": str(causes).lower()})

    async def chat(self, node_id: str, ip: str, message: str,
                   session_id: Optional[str] = None) -> EdgeResult:
        """
        L3 — ủy quyền câu hỏi cho agent của máy đó. Đắt: một lượt LLM ở edge.

        Timeout riêng và dài hơn hẳn: đo thật trên đội hình này là 4–20s, có lúc
        27s. Dùng chung EDGE_TIMEOUT (8s) thì mọi câu hỏi đều timeout.
        """
        body: Dict[str, Any] = {"message": message}
        if session_id:
            body["session_id"] = session_id
        return await self._authed(node_id, ip, settings.EDGE_AGENT_PORT,
                                  "/api/agent/chat", method="POST",
                                  timeout=settings.EDGE_CHAT_TIMEOUT, json=body)


client = EdgeClient()
