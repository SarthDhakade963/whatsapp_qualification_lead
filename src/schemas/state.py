from typing import List, Dict, Literal, Optional
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

class ConversationWorkflowState(BaseModel):
    # Entry
    input: InputPayload

    # Question processing
    questions: Questions

    # Answerable branch (optional for early exit)
    answerable_processing: Optional[AnswerableProcessing] = None

    # Skippable branch
    skippable_actions: Optional[SkippableActions] = None

    # Merge + post processing
    merged_output: Optional[MergedOutput] = None
    interaction_state: Optional[InteractionState] = None
    next_action: Optional[NextAction] = None

