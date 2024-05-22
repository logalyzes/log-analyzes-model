from pydantic import BaseModel


class Message(BaseModel):
    message: str
    isAnomal: int
    needAttention: int
    Level: int