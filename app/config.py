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
    NVIDIA_INFERENCE_TIMEOUT: float = float(os.environ.get("NVIDIA_INFERENCE_TIMEOUT", "85.0"))
    ROUTE_TIMEOUT: float = float(os.environ.get("ROUTE_TIMEOUT", "90.0"))

    # Shared secret for POST /assist authorization (empty string = auth disabled)
    ORANGE_API_KEY: str = os.environ.get("ORANGE_API_KEY", "").strip()

    # Temporary default model for Phase 6A only; capability-based selection deferred to Phase 6B/6C
    DEFAULT_NVIDIA_MODEL: str = os.environ.get("DEFAULT_NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    NVIDIA_CODER_MODEL: str = os.environ.get("NVIDIA_CODER_MODEL", "minimaxai/minimax-m3.1-flash")
    NVIDIA_BOLT_MODEL: str = os.environ.get("NVIDIA_BOLT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    NVIDIA_REASONING_MODEL: str = os.environ.get("NVIDIA_REASONING_MODEL", "nvidia/nemotron-3.5-bolt-e4b")

    @classmethod
    def get_nvidia_api_key(cls) -> str:
        """Resolve NVIDIA_API_KEY dynamically from environment or class settings."""
        return (os.environ.get("NVIDIA_API_KEY") or getattr(cls, "NVIDIA_API_KEY", "") or "").strip()

    # Hugging Face Inference Providers configuration (https://router.huggingface.co/v1)
    HF_TOKEN: str = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or ""
    HUGGINGFACE_BASE_URL: str = os.environ.get("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1")
    HUGGINGFACE_INFERENCE_TIMEOUT: float = float(os.environ.get("HUGGINGFACE_INFERENCE_TIMEOUT", "85.0"))
    DEFAULT_HUGGINGFACE_MODEL: str = os.environ.get("DEFAULT_HUGGINGFACE_MODEL", "Qwen/Qwen3.8-27B:fastest")

    @classmethod
    def get_hf_token(cls) -> str:
        """Resolve HF_TOKEN dynamically from environment or class settings."""
        return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or getattr(cls, "HF_TOKEN", "") or "").strip()

    # SambaNova Cloud configuration (https://api.sambanova.ai/v1)
    SAMBANOVA_API_KEY: str = os.environ.get("SAMBANOVA_API_KEY", "").strip()
    SAMBANOVA_BASE_URL: str = os.environ.get("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
    SAMBANOVA_INFERENCE_TIMEOUT: float = float(os.environ.get("SAMBANOVA_INFERENCE_TIMEOUT", "85.0"))
    DEFAULT_SAMBANOVA_MODEL: str = os.environ.get("DEFAULT_SAMBANOVA_MODEL", "gpt-oss-120b")
    SAMBANOVA_REASONING_MODEL: str = os.environ.get("SAMBANOVA_REASONING_MODEL", "Meta-Llama-3.3-70B-Instruct")

    @classmethod
    def get_sambanova_api_key(cls) -> str:
        """Resolve SAMBANOVA_API_KEY dynamically from environment or class settings."""
        return (os.environ.get("SAMBANOVA_API_KEY") or getattr(cls, "SAMBANOVA_API_KEY", "") or "").strip()

    # CORS configuration
    _raw_cors = os.environ.get("CORS_ORIGINS")
    CORS_ORIGINS: list[str] = (
        [o.strip() for o in _raw_cors.split(",") if o.strip()]
        if _raw_cors
        else [
            "https://orange-ai-steel.vercel.app",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )


settings = Settings()
