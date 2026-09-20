from typing import List, Dict, Any, Optional, Tuple
from app.providers import BaseProvider, get_provider
from app.services.intelligence_router import IntelligenceRouter, RouteDecision
from app.services.task_profile_builder import build_task_profile
from app.models.task_profile import TaskProfile
from app.models.schemas import AssistRequest


class LLMService:
    """
    Service layer for model inference requests.
    Delegates generation directly to the configured active AIProvider using the
    Intelligence Router for capability-based model selection.
    """

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        router: Optional[IntelligenceRouter] = None,
    ):
        self.provider = provider or get_provider()
        self.router = router or IntelligenceRouter()

    def generate_response(
        self,
        message: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        preferred_provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        task_profile: Optional[TaskProfile] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """
        Request model generation from the active provider.
        Routes task using the Intelligence Router.
        Returns (reply_text, model_identifier).
        """
        active_provider = get_provider(preferred_provider) if preferred_provider else self.provider

        # 1. Build or use TaskProfile
        profile = task_profile
        if profile is None:
            req = AssistRequest(
                message=message,
                model=model,
                task_type=task_type,
                modalities=modalities,
            )
            profile = build_task_profile(req)

        # 2. Evaluate Intelligence Router
        decision = self.router.route(profile)

        # 3. Dispatch to active provider with selected provider model ID
        reply = active_provider.generate(
            prompt=message,
            system_instruction=system_instruction,
            history=history,
            model=decision.provider_model_id,
            **kwargs,
        )
        return reply, f"{active_provider.name}:{decision.provider_model_id}"

    def generate(self, message: str, model: Optional[str] = None) -> str:
        """Helper for single message generation."""
        reply, _ = self.generate_response(message, model=model)
        return reply