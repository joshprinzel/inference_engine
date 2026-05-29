from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    total_text: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float