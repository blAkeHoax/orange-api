from pydantic import BaseModel
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