"""
task_profile_builder.py - Deterministic TaskProfile builder for Orange AI (Phase 6B).

Analyzes incoming AssistRequest metadata and prompt signals to construct a typed
TaskProfile in < 0.05ms without any LLM pre-classifier calls.
"""

import re
from typing import Set, List, Optional
from app.models.task_profile import TaskType, TaskProfile, LatencyPreference
from app.models.schemas import AssistRequest

_SUPPORTED_MODALITIES: Set[str] = {"text", "vision", "video"}

_CODING_PATTERNS = [
    r"```",                                      # Markdown code fences
    r"\bdef\s+[a-zA-Z_]",                        # Python function definition
    r"\bclass\s+[a-zA-Z_]",                      # Class definition
    r"\bimport\s+[a-zA-Z_]",                     # Import statement
    r"\bfrom\s+[a-zA-Z_]+\s+import",             # From import statement
    r"\bdebug\b",                                # Debugging
    r"\brefactor\b",                             # Refactoring
    r"\btraceback\b",                            # Python traceback
    r"\bsyntax\s+error\b",                       # Syntax error
    r"\bcompile\b",                              # Compilation
    r"\bwrite\s+(?:a\s+)?(?:code|script|function|program)\b",
    r"\bfix\s+(?:the\s+)?bug\b",
    r"\bpowershell\b",
    r"\bjavascript\b",
    r"\bpython\b",
    r"\bsql\b",
    r"\bregex\b",
    r"\balgorithm\b",
]

_COMPILED_CODING_REGEX = re.compile("|".join(_CODING_PATTERNS), re.IGNORECASE)


def normalize_modalities(modalities: Optional[List[str]]) -> frozenset[str]:
    """
    Normalize modality list to a frozenset.
    Defaults to {'text'} if empty or None.
    Rejects unsupported modalities cleanly with ValueError.
    """
    if not modalities:
        return frozenset({"text"})

    normalized: Set[str] = set()
    for m in modalities:
        clean = m.strip().lower()
        if not clean:
            continue
        if clean not in _SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality: '{clean}'. Supported modalities: {sorted(_SUPPORTED_MODALITIES)}"
            )
        normalized.add(clean)

    if not normalized:
        return frozenset({"text"})

    # Text is always included as a baseline modality
    normalized.add("text")
    return frozenset(normalized)


def build_task_profile(request: AssistRequest) -> TaskProfile:
    """
    Deterministically build a TaskProfile from an incoming AssistRequest.
    Validates task_type and modalities strictly.
    """
    # 1. Normalize and validate modalities
    modalities = normalize_modalities(request.modalities)

    # 2. Resolve TaskType
    resolved_task_type: TaskType
    requires_structured_output = False
    is_persistent_job = False

    if request.task_type is not None and request.task_type.strip():
        raw_type = request.task_type.strip().lower()
        try:
            resolved_task_type = TaskType(raw_type)
        except ValueError:
            valid_types = [t.value for t in TaskType]
            raise ValueError(
                f"Unsupported task_type: '{request.task_type}'. Valid task types: {valid_types}"
            )
    else:
        # Infer deterministically from modalities and prompt content
        if "video" in modalities:
            resolved_task_type = TaskType.VIDEO_ANALYSIS
        elif "vision" in modalities:
            resolved_task_type = TaskType.VISION_ANALYSIS
        elif _COMPILED_CODING_REGEX.search(request.message):
            resolved_task_type = TaskType.CODING
        else:
            resolved_task_type = TaskType.CONVERSATIONAL

    # 3. Formulate required capabilities based on resolved TaskType
    required_caps: Set[str] = set()
    preferred_caps: Set[str] = set()

    if resolved_task_type == TaskType.AGENT_DECISION:
        required_caps = {"text", "reasoning", "tool_use", "agentic"}
        requires_structured_output = True
        preferred_caps = {"coding"}

    elif resolved_task_type == TaskType.JOB_STEP:
        required_caps = {"text", "reasoning", "tool_use", "agentic"}
        requires_structured_output = True
        is_persistent_job = True
        preferred_caps = {"coding"}

    elif resolved_task_type == TaskType.CODING:
        required_caps = {"text", "reasoning", "coding"}
        preferred_caps = {"tool_use"}

    elif resolved_task_type == TaskType.VISION_ANALYSIS:
        required_caps = {"text", "vision"}
        if "vision" not in modalities:
            modalities = frozenset(set(modalities) | {"vision"})

    elif resolved_task_type == TaskType.VIDEO_ANALYSIS:
        required_caps = {"text", "video"}
        if "video" not in modalities:
            modalities = frozenset(set(modalities) | {"video"})

    else:  # TaskType.CONVERSATIONAL
        required_caps = {"text", "reasoning"}

    # Include non-text modalities in required capabilities
    for mod in modalities:
        if mod in ("vision", "video"):
            required_caps.add(mod)

    # 4. Resolve explicit model override from AssistRequest.model
    override_model = request.model.strip() if request.model and request.model.strip() else None

    return TaskProfile(
        task_type=resolved_task_type,
        required_capabilities=frozenset(required_caps),
        preferred_capabilities=frozenset(preferred_caps),
        modalities=modalities,
        requires_structured_output=requires_structured_output,
        urgency=LatencyPreference.NORMAL,
        override_model=override_model,
        is_persistent_job=is_persistent_job,
    )
