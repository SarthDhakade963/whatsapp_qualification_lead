from typing import List
from pydantic import BaseModel


class HandlerOutput(BaseModel):
    block_id: str
    facts: List[str]
    requires_confirmation: bool

