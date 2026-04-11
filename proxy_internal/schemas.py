"""Pydantic request/response schemas for the FastAPI routes."""

from typing import Optional, Union
from pydantic import BaseModel


class VisionContent(BaseModel):
    """Individual content item for vision messages (text or image_url)."""
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[VisionContent]]


class ChatRequest(BaseModel):
    # Pydantic V2 config: ``extra = "allow"`` lets unknown top-level sampling
    # params (top_p, frequency_penalty, stop, seed, …) pass through into
    # ``req.model_extra`` so _chat_completions_inner can forward them to
    # providers.  Keep this as ``model_config`` (V2 style), not the legacy
    # ``class Config`` which Pydantic V3 will remove.
    model_config = {"extra": "allow"}

    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
    stream: Optional[bool] = False
