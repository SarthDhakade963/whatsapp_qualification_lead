from typing import List, Dict, Literal, Optional, Annotated, TypedDict, Any
from pydantic import BaseModel, Field, ConfigDict


# =========================
# INPUT
# =========================

class InputPayload(BaseModel):
    raw_text: str


# =========================
# QUESTIONS
# =========================

QuestionClass = Literal[
    "ANSWERABLE",
    "FORBIDDEN",
    "MALFORMED",
    "HOSTILE"
]


class AtomicQuestion(BaseModel):
    id: str
    text: str


class ClassifiedQuestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    class_: QuestionClass = Field(alias="class")


class PartitionedQuestions(BaseModel):
    non_skippable: List[str] = []
    skippable: Dict[
        Literal["malformed", "forbidden", "hostile"],
        List[str]
    ] = {
        "malformed": [],
        "forbidden": [],
        "hostile": []
    }


class Questions(BaseModel):
    atomic: List[AtomicQuestion] = []
    classified: List[ClassifiedQuestion] = []
    partitioned: Optional[PartitionedQuestions] = None


# =========================
# ANSWERABLE PROCESSING
# =========================

QuestionCategory = Literal[
    "LOGISTICS",
    "COST",
    "ITINERARY",
    "POLICY"
]


class StructuredQuestion(BaseModel):
    id: str
    category: QuestionCategory
    text: str


TripConfidence = Literal["LOW", "MEDIUM", "HIGH"]


class TripContext(BaseModel):
    trip_id: str
    confidence: TripConfidence


AnswerStyle = Literal["HIGH_LEVEL", "DETAILED"]


HandlerName = Literal[
    "logistics_handler",
    "pricing_handler",
    "itinerary_handler"
]


class AnswerBlock(BaseModel):
    block_id: str
    question_ids: List[str]
    handler: HandlerName
    answer_style: AnswerStyle


class AnswerPlan(BaseModel):
    answer_blocks: List[AnswerBlock]


class HandlerOutput(BaseModel):
    block_id: str
    facts: List[str]
    requires_confirmation: bool


class AnswerableProcessing(BaseModel):
    structured_questions: List[StructuredQuestion] = []
    normalized_text: str
    trip_context: TripContext
    answer_plan: AnswerPlan
    handler_outputs: List[HandlerOutput] = []
    answer_text: Optional[str] = None


# =========================
# SKIPPABLE ACTIONS
# =========================

class SkippableActions(BaseModel):
    clarifications: List[str] = []
    boundaries: List[str] = []
    tone_safe_messages: List[str] = []


# =========================
# MERGED OUTPUT
# =========================

class MergedOutput(BaseModel):
    final_text: str


# =========================
# INTERACTION STATE
# =========================

DecisionStage = Literal[
    "EVALUATING",
    "ANSWERED",
    "ESCALATED"
]


class InteractionState(BaseModel):
    decision_stage: DecisionStage
    escalation_flag: bool


# =========================
# NEXT ACTION
# =========================

WorkflowType = Literal[
    "FOLLOW_UP",
    "CONVERGE",
    "HANDOFF",
    "END"
]


class NextAction(BaseModel):
    workflow: WorkflowType


# =========================
# ROOT STATE (LangGraph)
# =========================

def combine_handler_outputs(current: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer function to combine handler outputs from parallel execution."""
    if not current:
        return new if new else []
    if not new:
        return current
    # Combine lists of dict outputs
    return current + new


def combine_answerable_processing(
    current: Optional[AnswerableProcessing], 
    new: Optional[AnswerableProcessing]
) -> Optional[AnswerableProcessing]:
    """Reducer to merge answerable_processing from parallel handler execution.
    
    Handlers update the entire object, so we need to merge by:
    1. Combining handler_outputs lists (avoiding duplicates by block_id)
    2. Preferring new values for other fields (non-conflicting updates)
    """
    if not current:
        return new
    if not new:
        return current
    
    # Convert to dict if Pydantic models
    current_dict = current.model_dump() if hasattr(current, 'model_dump') else (current if isinstance(current, dict) else {})
    new_dict = new.model_dump() if hasattr(new, 'model_dump') else (new if isinstance(new, dict) else {})
    
    # Get handler_outputs from both
    current_outputs = current_dict.get("handler_outputs", []) or []
    new_outputs = new_dict.get("handler_outputs", []) or []
    
    # Combine outputs, avoiding duplicates by block_id
    seen_block_ids = set()
    combined_outputs = []
    
    # Add current outputs
    for output in current_outputs:
        block_id = output.get("block_id") if isinstance(output, dict) else getattr(output, "block_id", None)
        if block_id and block_id not in seen_block_ids:
            combined_outputs.append(output)
            seen_block_ids.add(block_id)
    
    # Add new outputs (only if not already present)
    for output in new_outputs:
        block_id = output.get("block_id") if isinstance(output, dict) else getattr(output, "block_id", None)
        if block_id and block_id not in seen_block_ids:
            combined_outputs.append(output)
            seen_block_ids.add(block_id)
        elif not block_id:
            # If no block_id, just add it (shouldn't happen but be safe)
            combined_outputs.append(output)
    
    # Merge dictionaries - prefer new for non-list fields
    merged_dict = current_dict.copy()
    merged_dict.update(new_dict)
    merged_dict["handler_outputs"] = combined_outputs
    
    # Return as dict (handlers work with dicts, LangGraph will handle conversion if needed)
    return merged_dict


class ConversationWorkflowState(TypedDict, total=False):
    """LangGraph state - TypedDict for conditional handler execution."""
    # Entry
    input: InputPayload

    # Question processing
    questions: Questions

    # Answerable branch (optional for early exit)
    # Annotated with reducer for parallel handler execution
    answerable_processing: Annotated[Optional[AnswerableProcessing], combine_answerable_processing]

    # Skippable branch
    skippable_actions: Optional[SkippableActions]

    # Merge + post processing
    merged_output: Optional[MergedOutput]
    interaction_state: Optional[InteractionState]
    next_action: Optional[NextAction]
    
    # Conversation history for context maintenance
    conversation_history: Optional[List[Dict[str, str]]]  # List of {"role": "user"/"assistant", "content": "..."}
    
    # Authoritative conversation state (decisions and metadata)
    conversation_state: Optional[Dict[str, Any]]  # Conversation state with focus, intent, risk, etc.
