"""
intelligence_router.py - Deterministic Capability-Based Intelligence Router (Phase 6B).

Selects the optimal model from the Model Registry based on a TaskProfile using a
unified 7-step deterministic hierarchy:
1. Explicit AssistRequest.model override validation
2. Required capability eligibility
3. Required modality eligibility
4. Task specialization
5. Preferred capability match
6. Structured-output preference
7. Static deterministic registry order
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from app.models.task_profile import TaskProfile, TaskType
from app.services.model_registry import (
    list_available_models,
    get_model,
    model_exists,
    ModelInfo,
    UnknownModelError,
)

logger = logging.getLogger("orange_api.router")


class RoutingError(Exception):
    """Base exception for all router failures."""
    pass


class NoEligibleModelError(RoutingError):
    """Raised when no registered model satisfies the required capabilities or modalities."""
    pass


class IneligibleOverrideError(RoutingError):
    """Raised when an explicitly requested model override lacks required capabilities."""
    pass


@dataclass(frozen=True)
class RouteDecision:
    """The immutable outcome of an Intelligence Router evaluation."""
    selected_model_key: str
    provider_model_id: str
    reason: str
    matched_capabilities: List[str]
    fallback_candidates: List[str]
    selection_step: str


# Task specialization preference order (internal model keys evaluated in order of preference)
_TASK_SPECIALIZATIONS: Dict[TaskType, List[str]] = {
    TaskType.CODING: ["sambanova_coder", "qwen_3_8_27b", "nvidia_coder", "nemotron_3_5_lightning_30b_a3b"],
    TaskType.REASONING: ["sambanova_reasoning", "gpt_oss_120b", "nvidia_reasoning", "nemotron_3_5_lightning_30b_a3b"],
    TaskType.AGENT_DECISION: ["nemotron_3_5_lightning_30b_a3b"],
    TaskType.JOB_STEP: ["nemotron_3_5_lightning_30b_a3b"],
    TaskType.VIDEO_ANALYSIS: ["gemma_4_31b_it"],
    TaskType.CONVERSATIONAL: ["nemotron_3_5_lightning_30b_a3b"],
    TaskType.GENERAL: ["nemotron_3_5_lightning_30b_a3b"],
}

# Canonical role specialization mapping (top preferred internal model key per task)
_CANONICAL_SPECIALIZATIONS: Dict[TaskType, str] = {
    TaskType.CODING: "sambanova_coder",
    TaskType.REASONING: "sambanova_reasoning",
    TaskType.AGENT_DECISION: "nemotron_3_5_lightning_30b_a3b",
    TaskType.JOB_STEP: "nemotron_3_5_lightning_30b_a3b",
    TaskType.VIDEO_ANALYSIS: "gemma_4_31b_it",
    TaskType.CONVERSATIONAL: "nemotron_3_5_lightning_30b_a3b",
    TaskType.GENERAL: "nemotron_3_5_lightning_30b_a3b",
}

# Static deterministic tie-break ordering across the foundation and legacy models
_STATIC_PRIORITY_ORDER: List[str] = [
    "nemotron_3_5_lightning_30b_a3b",
    "sambanova_coder",
    "sambanova_reasoning",
    "nvidia_coder",
    "nvidia_reasoning",
    "sambanova_gemma_4_31b",
    "sambanova_minimax_m3_1_flash",
    "sambanova_glm_5_3",
    "qwen_3_8_27b",
    "gpt_oss_120b",
    "gemma_4_31b_it",
    "gpt_oss_20b",
    "glm_5_3_flash",
]


class IntelligenceRouter:
    """
    Deterministic capability-based router for Orange AI.
    Maps task requirements to available models in the ModelRegistry.
    """

    def __init__(self, active_provider: Optional[Any] = None):
        self.active_provider = active_provider

    def _is_model_provider_available(self, model_info: ModelInfo, active_provider: Optional[Any] = None) -> bool:
        """
        Verify if the model's provider is currently configured and available.
        If a custom active_provider was explicitly injected into LLMService/Router,
        we verify against its provider name or fallback to the provider factory.
        """
        eff_provider = active_provider or self.active_provider
        if eff_provider is not None:
            prov_name = getattr(eff_provider, "name", None)
            if isinstance(prov_name, str) and prov_name != model_info.provider:
                try:
                    from app.providers.factory import get_provider
                    p = get_provider(model_info.provider)
                    return p.is_available()
                except Exception:
                    return False
            return True
        try:
            from app.providers.factory import get_provider
            p = get_provider(model_info.provider)
            return p.is_available()
        except Exception:
            return False

    def route(self, profile: TaskProfile, active_provider: Optional[Any] = None) -> RouteDecision:
        """
        Evaluate TaskProfile against available models using the 7-step hierarchy.
        Returns a RouteDecision detailing the chosen model, provider ID, and fallbacks.
        """
        eff_provider = active_provider or self.active_provider
        all_models = [
            m for m in list_available_models()
            if self._is_model_provider_available(m, eff_provider)
        ]

        # Helper to sort candidates deterministically by static priority order
        def sort_candidates(models: List[ModelInfo]) -> List[ModelInfo]:
            order_index = {key: i for i, key in enumerate(_STATIC_PRIORITY_ORDER)}
            return sorted(models, key=lambda m: order_index.get(m.key, 99))

        # ==========================================================
        # Step 1: Explicit AssistRequest.model Override Validation
        # ==========================================================
        if profile.override_model:
            override_key = profile.override_model
            if not model_exists(override_key):
                raise UnknownModelError(f"Unknown model identifier '{override_key}'.")

            model_info = get_model(override_key)

            # Validate model availability: reject unavailable models before provider inference
            if not getattr(model_info, "is_available", True):
                raise IneligibleOverrideError(
                    f"Model override '{model_info.key}' is currently unavailable."
                )

            # Validate provider availability: reject if upstream provider is unconfigured
            if not self._is_model_provider_available(model_info, eff_provider):
                raise IneligibleOverrideError(
                    f"Model override '{model_info.key}' provider '{model_info.provider}' is currently unconfigured or unavailable."
                )

            model_caps = set(model_info.capabilities)

            # Validate required capabilities
            missing_caps = profile.required_capabilities - model_caps
            if missing_caps:
                raise IneligibleOverrideError(
                    f"Model override '{model_info.key}' lacks required capabilities: {sorted(list(missing_caps))} "
                    f"for task type '{profile.task_type.value}'."
                )

            # Validate required modalities
            missing_mods = profile.modalities - model_caps
            if missing_mods:
                raise IneligibleOverrideError(
                    f"Model override '{model_info.key}' lacks required modalities: {sorted(list(missing_mods))}."
                )

            # Compute fallback candidates from other eligible models
            other_eligible = [
                m.key for m in sort_candidates(all_models)
                if m.key != model_info.key
                and profile.required_capabilities.issubset(set(m.capabilities))
                and profile.modalities.issubset(set(m.capabilities))
            ]

            decision = RouteDecision(
                selected_model_key=model_info.key,
                provider_model_id=model_info.provider_model_id,
                reason=f"Step 1: Explicit AssistRequest.model Override ('{model_info.key}')",
                matched_capabilities=sorted(list(profile.required_capabilities)),
                fallback_candidates=other_eligible,
                selection_step="EXPLICIT_OVERRIDE",
            )
            self._log_decision(decision, profile)
            return decision

        # ==========================================================
        # Step 2: Required Capability Eligibility
        # ==========================================================
        candidates = [
            m for m in all_models
            if profile.required_capabilities.issubset(set(m.capabilities))
        ]
        if not candidates:
            raise NoEligibleModelError(
                f"No registered models satisfy required capabilities: {sorted(list(profile.required_capabilities))}."
            )

        # ==========================================================
        # Step 3: Required Modality Eligibility
        # ==========================================================
        candidates = [
            m for m in candidates
            if profile.modalities.issubset(set(m.capabilities))
        ]
        if not candidates:
            raise NoEligibleModelError(
                f"No registered models satisfy required modalities: {sorted(list(profile.modalities))}."
            )

        # If exactly one model satisfies hard requirements for non-text modalities, select it immediately
        if len(candidates) == 1 and (profile.modalities - {"text"}):
            selected = candidates[0]
            decision = RouteDecision(
                selected_model_key=selected.key,
                provider_model_id=selected.provider_model_id,
                reason="Step 3: Sole Eligible Candidate for Required Modalities and Capabilities",
                matched_capabilities=sorted(list(profile.required_capabilities | profile.modalities)),
                fallback_candidates=[],
                selection_step="SOLE_ELIGIBLE_MODALITY",
            )
            self._log_decision(decision, profile)
            return decision

        sorted_candidates = sort_candidates(candidates)

        def make_decision(chosen: ModelInfo, reason: str, step: str) -> RouteDecision:
            fallbacks = [m.key for m in sorted_candidates if m.key != chosen.key]
            d = RouteDecision(
                selected_model_key=chosen.key,
                provider_model_id=chosen.provider_model_id,
                reason=reason,
                matched_capabilities=sorted(list(set(chosen.capabilities) & (profile.required_capabilities | profile.preferred_capabilities | profile.modalities))),
                fallback_candidates=fallbacks,
                selection_step=step,
            )
            self._log_decision(d, profile)
            return d

        # ==========================================================
        # Step 4: Task Specialization Match
        # ==========================================================
        # If task requires structured output and is conversational/general,
        # skip conversational default in Step 4 so Step 6 structured-output preference applies.
        skip_step4_specialization = (
            profile.requires_structured_output
            and profile.task_type in (TaskType.CONVERSATIONAL, TaskType.GENERAL)
        )
        if not skip_step4_specialization:
            specialized_keys = _TASK_SPECIALIZATIONS.get(profile.task_type, [])
            for specialized_key in specialized_keys:
                matching = [m for m in sorted_candidates if m.key == specialized_key]
                if matching:
                    return make_decision(
                        chosen=matching[0],
                        reason=f"Step 4: Task Specialization Match for '{profile.task_type.value}'",
                        step="TASK_SPECIALIZATION",
                    )

        # ==========================================================
        # Step 5: Preferred Capability Match
        # ==========================================================
        if profile.preferred_capabilities:
            scored = [
                (len(set(m.capabilities) & profile.preferred_capabilities), m)
                for m in sorted_candidates
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_score, top_model = scored[0]
            if top_score > 0 and (len(scored) == 1 or top_score > scored[1][0]):
                return make_decision(
                    chosen=top_model,
                    reason=f"Step 5: Preferred Capability Match ({top_score} capability matches)",
                    step="PREFERRED_CAPABILITY",
                )

        # ==========================================================
        # Step 6: Structured-Output Preference
        # ==========================================================
        if profile.requires_structured_output:
            structured_order = [
                "sambanova_coder",
                "qwen_3_8_27b",
                "nvidia_coder",
                "nemotron_3_5_lightning_30b_a3b",
                "gpt_oss_120b",
                "sambanova_reasoning",
                "nvidia_reasoning",
                "sambanova_gemma_4_31b",
                "sambanova_minimax_m3_1_flash",
                "sambanova_glm_5_3",
                "gemma_4_31b_it",
                "glm_5_3_flash",
            ]
            for key in structured_order:
                matching = [m for m in sorted_candidates if m.key == key]
                if matching:
                    return make_decision(
                        chosen=matching[0],
                        reason="Step 6: Structured-Output Agentic Precedence",
                        step="STRUCTURED_OUTPUT_PREFERENCE",
                    )

        # ==========================================================
        # Step 7: Static Deterministic Registry Order
        # ==========================================================
        return make_decision(
            chosen=sorted_candidates[0],
            reason="Step 7: Static Deterministic Registry Priority",
            step="STATIC_REGISTRY_ORDER",
        )

    def _log_decision(self, decision: RouteDecision, profile: TaskProfile) -> None:
        logger.info(
            "[ROUTER] Task: %s -> Model: '%s' (%s) | Step: %s | Reason: %s",
            profile.task_type.value,
            decision.selected_model_key,
            decision.provider_model_id,
            decision.selection_step,
            decision.reason,
        )
