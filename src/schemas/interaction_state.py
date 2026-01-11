from typing import Literal
from pydantic import BaseModel


DecisionStage = Literal[
    "EVALUATING",
    "ANSWERED",
    "ESCALATED"
]


class InteractionState(BaseModel):
    decision_stage: DecisionStage
    escalation_flag: bool


WorkflowType = Literal[
    "FOLLOW_UP",
    "CONVERGE",
    "HANDOFF",
    "END"
]


class NextAction(BaseModel):
    workflow: WorkflowType

