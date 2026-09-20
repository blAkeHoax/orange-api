from typing import Optional, List, Dict, Any
from app.services.llm_service import LLMService
from app.models.schemas import Message


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Orange AI, an intelligent, helpful, and concise personal AI assistant."
)


class AssistService:
    """
    Business logic service for the /assist endpoint.
    Prepares history and context before dispatching to LLMService.
    """

    def __init__(self):
        self.llm = LLMService()

    def assist(
        self,
        message: str,
        history: Optional[List[Message]] = None,
        profile: Optional[str] = None,
        system_instruction: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        modalities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process an assistant query and return structured response data.
        Combines system_instruction and user profile into the model context.
        """
        history_dicts = None
        if history:
            history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]

        full_instruction = (
            system_instruction.strip()
            if system_instruction and system_instruction.strip()
            else DEFAULT_SYSTEM_INSTRUCTION
        )

        if profile and profile.strip():
            profile_block = f"User Profile & Background:\n{profile.strip()}"
            if full_instruction:
                full_instruction = f"{full_instruction}\n\n{profile_block}"
            else:
                full_instruction = profile_block

        reply, model_name = self.llm.generate_response(
            message=message,
            system_instruction=full_instruction,
            history=history_dicts,
            preferred_provider=preferred_provider,
            model=model,
            task_type=task_type,
            modalities=modalities,
        )

        return {
            "reply": reply,
            "model": model_name,
            "success": True,
        }