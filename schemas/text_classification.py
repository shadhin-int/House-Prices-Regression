from pydantic import BaseModel


class TextRequest(BaseModel):
	text: str


class TextResponse(BaseModel):
	prediction: int
	confidence: float
