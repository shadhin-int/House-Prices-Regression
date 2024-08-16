from pydantic import BaseModel


class TextRequest(BaseModel):
	text: str


class TextResponse(BaseModel):
	class_name: str
	confidence: float
