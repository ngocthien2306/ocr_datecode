from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"
    # TTL retention (days). MongoDB auto-deletes documents older than this.
    # 0 disables the TTL index for that collection.
    INFERENCE_RESULTS_TTL_DAYS: int = 30
    ACTION_LOGS_TTL_DAYS: int = 90

    # JWT
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 360

    # CORS
    CORS_ORIGINS: List[str] = [
        # "http://localhost:3000", 
        # "http://localhost:5173",
        # "http://localhost:5174",
        # "https://suntech-vision-api.ngrok.app",
        # "https://suntech-vision.ngrok.app",
        "*"  # Allow all origins in development
    ]

    # Application
    APP_NAME: str = "OCR Datecode API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API Base URL (for generating full URLs in responses)
    # Default to localhost, override with env var for production/ngrok
    API_BASE_URL: str = "http://localhost:8000"

    # Timezone used for storing/displaying datetimes (IANA zone name)
    # Default to Vietnam (Ho Chi Minh) as requested; change via .env if needed
    TIMEZONE: str = "Asia/Ho_Chi_Minh"

    OPENAI_API_KEY: str = ""
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
