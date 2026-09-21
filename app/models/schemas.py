from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class AssistRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    profile: Optional[str] = ""
    system_instruction: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None              # Sole external model override
    task_type: Optional[str] = None          # Explicit task archetype hint
    modalities: Optional[List[str]] = None   # Minimal routing-only modality signal


class AssistResponse(BaseModel):
    reply: str
    model: str
    success: bool


class ImageGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    model: Optional[str] = "flux_2_klein_4b"
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    steps: Optional[int] = Field(default=4, ge=1, le=4)
    seed: Optional[int] = Field(default=None, ge=0)
    samples: Optional[int] = Field(default=1, ge=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return v

    @field_validator("width")
    @classmethod
    def validate_width(cls, v: Optional[int]) -> int:
        if v is not None and v != 1024:
            raise ValueError(f"Unsupported width ({v}). Currently only 1024 is supported.")
        return v or 1024

    @field_validator("height")
    @classmethod
    def validate_height(cls, v: Optional[int]) -> int:
        if v is not None and v != 1024:
            raise ValueError(f"Unsupported height ({v}). Currently only 1024 is supported.")
        return v or 1024


class GeneratedImage(BaseModel):
    b64_json: str
    mime_type: str = "image/png"
    width: int = 1024
    height: int = 1024


class ImageGenerateResponse(BaseModel):
    success: bool
    model: str
    images: List[GeneratedImage]
    seed: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None