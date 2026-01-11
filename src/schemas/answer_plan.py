from typing import List, Literal
from pydantic import BaseModel


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

