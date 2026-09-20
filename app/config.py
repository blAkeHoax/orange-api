import os
from pathlib import Path
from dotenv import load_dotenv

# Path resolution
API_DIR = Path(__file__).resolve().parent.parent

# Load environment: Orange API owns its credentials in Orange API/.env exclusively
api_env = API_DIR / ".env"
if api_env.exists():
    load_dotenv(api_env)


class Settings:
    """Application configuration and environment settings."""
    PROJECT_NAME: str = "Orange API"
    VERSION: str = "1.0.0"

    # AI Provider configuration: NVIDIA is the active provider
    AI_PROVIDER: str = os.environ.get("AI_PROVIDER", os.environ.get("DEFAULT_PROVIDER", "nvidia")).lower()
    DEFAULT_PROVIDER: str = AI_PROVIDER  # Backwards compatibility alias

    # NVIDIA configuration
    NVIDIA_API_KEY: str | None = os.environ.get("NVIDIA_API_KEY")
    NVIDIA_BASE_URL: str = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    # Temporary default model for Phase 6A only; capability-based selection deferred to Phase 6B/6C
    DEFAULT_NVIDIA_MODEL: str = os.environ.get("DEFAULT_NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")


settings = Settings()
