"""
LLM factory.

Một chỗ duy nhất tạo ChatOpenAI. Lý do tồn tại: ChatOpenAI mặc định đọc
api_key từ os.environ["OPENAI_API_KEY"], nhưng pydantic-settings chỉ nạp .env
vào `settings` chứ KHÔNG export ra os.environ — nên nếu không truyền api_key
tường minh thì agent chỉ chạy được khi ai đó nhớ `export OPENAI_API_KEY`.
"""

from langchain_openai import ChatOpenAI

from agent_app.core.config import settings


def make_llm(model: str | None = None, temperature: float = 0.7, **kwargs) -> ChatOpenAI:
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY chưa được cấu hình trong agent_service/.env"
        )

    return ChatOpenAI(
        model=model or settings.DEFAULT_MODEL,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
        **kwargs,
    )
