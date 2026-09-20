"""
model_registry.py - Model Registry Foundation for Orange AI (Phase 6A).

Defines the four-model foundation hosted via NVIDIA's OpenAI-compatible API:
1. GPT-OSS-20B (openai/gpt-oss-20b)
2. Nemotron-3.5-Lightning-30B-A3B (nvidia/nemotron-3.5-lightning-30b-a3b)
3. GLM-5-3-Flash (z-ai/glm-5.3-flash)
4. Gemma 4 31B IT (google/gemma-4-31b-it)

Provides deterministic registry lookup APIs and capability metadata.
Automatic routing, task classification, and scoring are strictly deferred
to Phase 6B/6C.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


class UnknownModelError(ValueError):
    """Raised when a requested model key or ID is not found in the registry."""
    pass


@dataclass(frozen=True)
class ModelInfo:
    """
    Metadata representation of an AI model in the registry.
    """
    key: str
    provider_model_id: str
    provider: str
    display_name: str
    capabilities: List[str]
    description: Optional[str] = None


# Locked Four-Model Foundation
_MODELS: Dict[str, ModelInfo] = {
    "gpt_oss_20b": ModelInfo(
        key="gpt_oss_20b",
        provider_model_id="openai/gpt-oss-20b",
        provider="nvidia",
        display_name="GPT-OSS-20B",
        capabilities=["text", "reasoning", "coding"],
        description="Open-weight language model optimized for text, reasoning, and code generation.",
    ),
    "nemotron_3_5_lightning_30b_a3b": ModelInfo(
        key="nemotron_3_5_lightning_30b_a3b",
        provider_model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
        provider="nvidia",
        display_name="Nemotron-3.5-Lightning-30B-A3B",
        capabilities=["text", "reasoning", "coding", "tool_use", "agentic"],
        description="Fast and capable agentic model optimized for structured reasoning and tool execution.",
    ),
    "glm_5_3_flash": ModelInfo(
        key="glm_5_3_flash",
        provider_model_id="z-ai/glm-5.3-flash",
        provider="nvidia",
        display_name="GLM-5-3-Flash",
        capabilities=["text", "reasoning", "tool_use", "agentic", "vision"],
        description="Low-latency multimodal model with vision, tool use, and agentic capabilities.",
    ),
    "gemma_4_31b_it": ModelInfo(
        key="gemma_4_31b_it",
        provider_model_id="google/gemma-4-31b-it",
        provider="nvidia",
        display_name="Gemma 4 31B IT",
        capabilities=["text", "reasoning", "coding", "tool_use", "agentic", "vision", "video"],
        description="High-capacity multimodal instruction-tuned model supporting text, vision, and video.",
    ),
}

# Reverse lookup mapping from provider_model_id to ModelInfo
_PROVIDER_ID_MAP: Dict[str, ModelInfo] = {
    model.provider_model_id: model for model in _MODELS.values()
}


def get_model(model_key_or_id: str) -> ModelInfo:
    """
    Look up a model by internal key (e.g. 'gemma_4_31b_it') or by
    provider model ID (e.g. 'google/gemma-4-31b-it').
    Raises UnknownModelError if the model is unrecognized.
    """
    cleaned = model_key_or_id.strip()
    if cleaned in _MODELS:
        return _MODELS[cleaned]
    if cleaned in _PROVIDER_ID_MAP:
        return _PROVIDER_ID_MAP[cleaned]

    known_keys = list(_MODELS.keys())
    known_ids = list(_PROVIDER_ID_MAP.keys())
    raise UnknownModelError(
        f"Unknown model identifier '{model_key_or_id}'. "
        f"Registered model keys: {known_keys}, provider IDs: {known_ids}"
    )


def get_model_by_provider_id(model_id: str) -> ModelInfo:
    """Look up a model strictly by its provider model ID."""
    cleaned = model_id.strip()
    if cleaned in _PROVIDER_ID_MAP:
        return _PROVIDER_ID_MAP[cleaned]
    raise UnknownModelError(
        f"Unknown provider model ID '{model_id}'. Available: {list(_PROVIDER_ID_MAP.keys())}"
    )


def list_models() -> List[ModelInfo]:
    """Return all registered models."""
    return list(_MODELS.values())


def list_available_models() -> List[ModelInfo]:
    """
    Return all models currently considered available in the registry.
    In Phase 6A, all four models in the foundation are defined as available.
    """
    return list(_MODELS.values())


def model_exists(model_key_or_id: str) -> bool:
    """Check if a model exists by internal key or provider model ID."""
    cleaned = model_key_or_id.strip()
    return cleaned in _MODELS or cleaned in _PROVIDER_ID_MAP


def resolve_provider_model_id(model_key_or_id: Optional[str], default_id: str) -> str:
    """
    Resolve a given model key or provider model ID into the canonical
    provider model ID to send to the provider API.
    If model_key_or_id is None or empty, returns default_id.
    """
    if not model_key_or_id:
        return default_id
    model = get_model(model_key_or_id)
    return model.provider_model_id
