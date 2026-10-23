"""
task_profile.py - Structured task representation for the Intelligence Router (Phase 6B).

Defines TaskType, LatencyPreference, and TaskProfile dataclass.
Encapsulates task archetypes, capability requirements, and modality constraints.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import FrozenSet, Optional


class TaskType(str, Enum):
    """
    Categorical archetype of a computational task for model specialization matching.
    """
    CONVERSATIONAL = "conversational"
    CODING = "coding"
    AGENT_DECISION = "agent_decision"
    JOB_STEP = "job_step"
    VIDEO_ANALYSIS = "video_analysis"


class LatencyPreference(str, Enum):
    """
    Latency sensitivity hint. Retained for contract completeness; inactive in Phase 6B.
    """
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"


@dataclass(frozen=True)
class TaskProfile:
    """
    Immutable representation of a task's computational and capability requirements.
    """
    task_type: TaskType
    required_capabilities: FrozenSet[str]
    preferred_capabilities: FrozenSet[str] = field(default_factory=frozenset)
    modalities: FrozenSet[str] = field(default_factory=lambda: frozenset({"text"}))
    requires_structured_output: bool = False
    urgency: LatencyPreference = LatencyPreference.NORMAL
    override_model: Optional[str] = None  # Copied from AssistRequest.model
    is_persistent_job: bool = False
