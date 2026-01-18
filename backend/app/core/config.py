from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "ocr_datecode_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL_RECENT: int = 60  # 1 minute for recent data
    REDIS_CACHE_TTL_HISTORICAL: int = 300  # 5 minutes for historical data

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

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # API Base URL (for generating full URLs in responses)
    # Default to localhost, override with env var for production/ngrok
    API_BASE_URL: str = "http://localhost:8000"

    # Timezone used for storing/displaying datetimes (IANA zone name)
    # Default to Vietnam (Ho Chi Minh) as requested; change via .env if needed
    TIMEZONE: str = "Asia/Ho_Chi_Minh"

    # Storage Paths
    BACKEND_BASE_PATH: str = "/home/demo/Source/ocr_datecode/backend"
    UPLOADS_BASE_PATH: str = "/home/demo/Source/ocr_datecode/backend/uploads"
    INFERENCE_RESULTS_PATH: str = "/home/demo/Source/ocr_datecode/backend/uploads/inference_results"
    SAVE_FRAME_PATH: str = "/home/demo/Source/ocr_datecode/backend/uploads/save_frame"
    TEMPLATE_UPLOAD_PATH: str = "/home/demo/Source/ocr_datecode/backend/uploads/templates"
    TEMPLATE_VISUALIZE_PATH: str = "/home/demo/Source/ocr_datecode/backend/uploads/visualizations"
    AVATAR_UPLOAD_PATH: str = "/home/demo/Source/ocr_datecode/backend/uploads/avatar"
    LOGS_PATH: str = "/home/demo/Source/ocr_datecode/backend/logs"
    CAMERA_SETTINGS_PATH: str = "/home/demo/Source/ocr_datecode/backend/camera_settings"
    INFERENCE_STATE_PATH: str = "/home/demo/Source/ocr_datecode/backend/inference_state"

    # AI Services
    AI_SERVICES_PATH: str = "/home/demo/Source/ocr_datecode/ai_services"
    CAMERA_SCRIPT_PATH: str = "/home/demo/Source/ocr_datecode/ai_services/camera_shm_producer.py"

    # Image Processing
    FRAME_SAVE_QUALITY: int = 95

    # Trigger Configuration
    TRIGGER_FILE_PATH: str = "/tmp"

    # OpenAI
    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
