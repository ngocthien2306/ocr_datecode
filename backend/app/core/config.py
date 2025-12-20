from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"

    # JWT
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

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

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
