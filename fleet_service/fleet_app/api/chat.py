"""Chat với `fleet_orchestrator`."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fleet_app.agents import fleet_orchestrator
from fleet_app.core.config import settings

router = APIRouter(prefix="/api/fleet", tags=["Fleet Agent"])


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


@router.post("/chat", summary="Hỏi trợ lý cấp đội hình")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    if not settings.OPENAI_API_KEY:
        raise HTTPException(503, "Chưa cấu hình OPENAI_API_KEY cho fleet service")
    try:
        return await fleet_orchestrator.run(req.message, req.history)
    except Exception as e:
        # Trả lỗi đọc được thay vì 500 trắng: hết credit OpenAI là tình huống đã
        # xảy ra thật, và lúc đó người dùng cần biết nguyên nhân chứ không phải
        # một trang lỗi trống.
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e
