"""
Sổ đăng ký máy: phát hiện qua tailnet, rồi dò xem mỗi máy làm được gì.

**Vì sao phát hiện qua Tailscale thay vì khai báo trong file.** Tailnet đã là
nguồn sự thật cho câu hỏi "có những máy nào". Máy mới vào tailnet và chạy agent
service là tự xuất hiện ở vòng quét kế tiếp — không sửa file, không restart, không
deploy. File `config/machines.json` vẫn có, nhưng chỉ để **đặt tên đẹp và gắn
nhãn**, không phải để khai báo sự tồn tại. Máy không có trong file vẫn được quản,
chỉ là hiện tên hostname thô.

**Vì sao khoá theo Tailscale node id.** Trên đội hình này có BỐN máy cùng
hostname `suntech-desktop`; IP thì đổi khi máy rời/vào lại tailnet. Node id là
thứ duy nhất vừa bền vừa duy nhất, nên lịch sử nối đúng máy qua thời gian.

**Thang năng lực.** Fleet không giả định máy hỗ trợ gì — nó dò từ dưới lên và
dừng ở bậc cao nhất máy đó đỡ được:

    L0  cổng 8100 mở                 → máy tồn tại
    L1  GET /api/agent/health         → sống, mấy agent, backend có với tới không
    L2  login + GET /api/agent/agents → hỏi được loại câu nào
    L3  POST /api/agent/chat          → ủy quyền được (không dò, vì tốn tiền)
    L4  /api/jetson-monitoring/metrics→ số liệu phần cứng

Nhờ vậy fleet quản được cả máy `release_v1` lẫn `release_v2`, và máy chưa kịp
nâng cấp thì tự động đi đường thấp hơn thay vì gãy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from fleet_app.core.config import MACHINES_FILE, settings
from fleet_app.core.edge_client import client

logger = logging.getLogger(__name__)


@dataclass
class Machine:
    node_id: str
    hostname: str
    ip: str
    online: bool = True

    # nhãn người đặt, từ config/machines.json
    label: Optional[str] = None
    line: Optional[str] = None
    note: Optional[str] = None
    # Model thiết bị là thuộc tính TĨNH của máy, nên nằm ở đây chứ không lấy từ
    # số liệu chạy. `nvp_model` mà backend trả về là CHẾ ĐỘ ĐIỆN (MAXN_SUPER),
    # không phải model — hai thứ khác nhau, gộp lại thì cột "Thiết bị" hiện ra
    # tên một chế độ nguồn.
    model: Optional[str] = None
    # Chặn chế độ điện cho máy mà `nvp_model` đã bị ghi đè thành chuỗi khác.
    power_mode_override: Optional[str] = None
    # Vị trí trên sơ đồ mặt bằng: {x, y, rotation, zone}. Để trong config chứ
    # không hardcode trong component — đổi mặt bằng thì sửa JSON, không sửa code.
    floor: Optional[Dict[str, Any]] = None

    # kết quả dò năng lực
    level: int = 0
    agents: List[str] = field(default_factory=list)
    agent_status: Optional[str] = None
    backend_reachable: Optional[bool] = None
    has_metrics: bool = False

    last_seen: Optional[float] = None       # lần cuối đạt tối thiểu L1
    last_error: Optional[str] = None
    probe_ms: Optional[int] = None

    @property
    def name(self) -> str:
        """Tên hiển thị: nhãn người đặt, không có thì hostname."""
        return self.label or self.hostname

    @property
    def stale(self) -> bool:
        if self.last_seen is None:
            return True
        return (time.time() - self.last_seen) > settings.STALE_AFTER

    def state(self) -> str:
        """
        Trạng thái gộp, phân biệt ba tình huống mà gộp lại là giấu mất sự cố.

        `agent_down` khác `unreachable` ở chỗ máy VẪN ĐANG SẢN XUẤT, chỉ là agent
        tắt — trên đội hình này agent chạy uvicorn trần nên reboot là mất, và đó
        là tình huống thường gặp chứ không phải ngoại lệ.
        """
        if not self.online:
            return "offline"          # tailnet nói máy không online
        if self.level >= 1 and not self.stale:
            return "ok"
        if self.level == 0 and self.last_seen:
            return "agent_down"       # từng thấy agent, giờ cổng không trả lời
        return "unreachable"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name
        d["state"] = self.state()
        d["stale"] = self.stale
        return d


class Registry:
    def __init__(self) -> None:
        self._machines: Dict[str, Machine] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        self._lock = asyncio.Lock()

    # --- nhãn --------------------------------------------------------------

    def load_labels(self) -> None:
        """
        Nhãn do người đặt. Thiếu file thì bỏ qua — nhãn là thứ làm đẹp, không
        phải điều kiện để một máy được quản lý.
        """
        try:
            raw = json.loads(MACHINES_FILE.read_text(encoding="utf-8"))
            self._labels = {k: v for k, v in raw.items() if isinstance(v, dict)}
            logger.info("Đọc nhãn cho %d máy từ %s", len(self._labels), MACHINES_FILE)
        except FileNotFoundError:
            self._labels = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Không đọc được %s: %s — bỏ qua nhãn", MACHINES_FILE, e)
            self._labels = {}

    # --- phát hiện ---------------------------------------------------------

    def _tailnet_peers(self) -> List[Dict[str, Any]]:
        try:
            out = subprocess.run([settings.TAILSCALE_BIN, "status", "--json"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                logger.warning("tailscale status lỗi: %s", out.stderr.strip()[:200])
                return []
            data = json.loads(out.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning("Không chạy được tailscale (%s): %s", settings.TAILSCALE_BIN, e)
            return []

        peers = []
        for p in (data.get("Peer") or {}).values():
            ips = p.get("TailscaleIPs") or []
            if not ips:
                continue
            peers.append({
                "node_id": p.get("ID") or ips[0],
                "hostname": p.get("HostName") or ips[0],
                "ip": ips[0],
                "online": bool(p.get("Online")),
            })
        # Chính máy đang chạy fleet cũng nằm trên tailnet nhưng không phải thiết
        # bị dây chuyền — bỏ qua, nếu không dashboard sẽ có một ô "không với tới
        # được" vĩnh viễn trỏ vào chính nó.
        return peers

    # --- dò năng lực -------------------------------------------------------

    async def probe(self, m: Machine) -> Machine:
        t0 = time.monotonic()
        m.last_error = None

        health = await client.health(m.ip)
        if not health.ok:
            m.level = 0
            m.last_error = health.error
            m.probe_ms = int((time.monotonic() - t0) * 1000)
            return m

        h = health.data if isinstance(health.data, dict) else {}
        m.level = 1
        m.agent_status = h.get("status")
        m.backend_reachable = h.get("backend_reachable")
        m.last_seen = time.time()

        ag = await client.agents(m.node_id, m.ip)
        if ag.ok and isinstance(ag.data, list):
            m.level = 2
            m.agents = [a.get("agent_id") for a in ag.data if a.get("agent_id")]
        else:
            m.last_error = ag.error

        # L4 dò bằng một lời gọi thật thay vì suy đoán từ phiên bản: máy có thể
        # có endpoint mà vòng giám sát bên trong đã chết, và hai thứ đó khác nhau.
        met = await client.system_metrics(m.ip)
        m.has_metrics = met.ok and isinstance(met.data, dict) and bool(met.data.get("data"))
        if m.has_metrics:
            m.level = 4

        m.probe_ms = int((time.monotonic() - t0) * 1000)
        return m

    # --- vòng làm mới ------------------------------------------------------

    async def refresh(self) -> List[Machine]:
        peers = self._tailnet_peers()
        async with self._lock:
            seen = set()
            for p in peers:
                nid = p["node_id"]
                seen.add(nid)
                m = self._machines.get(nid)
                if m is None:
                    m = Machine(node_id=nid, hostname=p["hostname"], ip=p["ip"])
                    self._machines[nid] = m
                    logger.info("Phát hiện máy mới trên tailnet: %s (%s)", m.hostname, m.ip)
                m.hostname, m.ip, m.online = p["hostname"], p["ip"], p["online"]
                lbl = self._labels.get(nid) or {}
                m.label, m.line, m.note = lbl.get("label"), lbl.get("line"), lbl.get("note")
                m.model = lbl.get("model")
                m.power_mode_override = lbl.get("power_mode")
                m.floor = lbl.get("floor")
            # Máy rời tailnet: giữ lại bản ghi và đánh dấu offline thay vì xoá —
            # xoá đi thì mất luôn `last_seen`, tức mất câu trả lời "im từ bao giờ".
            for nid, m in self._machines.items():
                if nid not in seen:
                    m.online = False

        targets = [m for m in self._machines.values() if m.online]
        if targets:
            # Cùng lý do với fan_out: dò 50 máy cùng lúc là tự bóp nghẹt link.
            sem = asyncio.Semaphore(settings.FANOUT_CONCURRENCY)

            async def _probe(m: Machine) -> None:
                async with sem:
                    await self.probe(m)

            await asyncio.gather(*(_probe(m) for m in targets))
        return self.all()

    def all(self) -> List[Machine]:
        # Máy chạy được xếp trước, rồi tới tên — để mắt nhìn xuống danh sách thì
        # thứ đang hỏng không lẫn vào giữa.
        return sorted(self._machines.values(),
                      key=lambda m: (m.state() != "ok", m.name.lower()))

    def get(self, key: str) -> Optional[Machine]:
        """
        Tìm theo định danh DUY NHẤT: node id, nhãn, hoặc IP.

        Hostname cố ý KHÔNG nằm trong đây, dù người dùng có thể gõ nó. Trên đội
        hình này bốn máy cùng mang hostname `suntech-desktop`, nên khớp theo
        hostname rồi trả về cái đầu tiên là đưa số liệu của máy khác mà người
        dùng không hề biết. Hostname được xử lý ở `resolve()`, nơi mọi ứng viên
        đều được trả về để hỏi lại.
        """
        if key in self._machines:
            return self._machines[key]
        k = key.strip().lower()
        for m in self._machines.values():
            if k and k in {(m.label or "").lower(), m.ip}:
                return m
        return None

    def resolve(self, key: str) -> List[Machine]:
        """
        Khớp mờ, trả về TẤT CẢ ứng viên.

        Trả danh sách chứ không tự chọn cái đầu tiên: bốn máy ở đây trùng hostname
        `suntech-desktop`, nên đoán bừa nghĩa là người dùng nhận số liệu của máy
        khác mà không hề biết. Mơ hồ thì phải hỏi lại.
        """
        exact = self.get(key)
        if exact:
            return [exact]
        k = key.strip().lower()
        return [m for m in self._machines.values()
                if k and (k in m.name.lower() or k in m.hostname.lower() or k in m.ip)]


registry = Registry()
