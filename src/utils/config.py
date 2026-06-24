import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # API Keys & Env
    GEMINI_API_KEY: str = ""
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/analysis/resume_analyzer.db"
    
    # Storage Paths
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")
    ANALYSIS_DIR: str = str(BASE_DIR / "data" / "analysis")
    
    # Model config
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Centralized Gemini API Configuration (Single Source of Truth)
try:
    import google.generativeai as genai
    if settings.GEMINI_API_KEY:
        # Filter mock keys or empty strings
        key = settings.GEMINI_API_KEY
        if key and not key.startswith("your_") and len(key) > 8:
            genai.configure(api_key=key)
            # We don't log the actual key for security reasons, but we log successful configuration.
            print("Gemini API successfully configured globally via settings.")
        else:
            print("Gemini API key is a mock or empty. Skipping global configuration.")
except ImportError:
    print("google-generativeai package not installed. Gemini features will be disabled.")
except Exception as e:
    print(f"Failed to configure Gemini API globally: {e}")

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.ANALYSIS_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(settings.ANALYSIS_DIR) / "app.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("ai_resume_analyzer")
logger.info(f"Initialized application config. Base Dir: {BASE_DIR}")
logger.info(f"Database URL: {settings.DATABASE_URL}")
logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
