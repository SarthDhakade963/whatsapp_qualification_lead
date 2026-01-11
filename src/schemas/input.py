from pydantic import BaseModel


class InputPayload(BaseModel):
    raw_text: str

