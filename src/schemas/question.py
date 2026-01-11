from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


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

