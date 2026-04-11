"""Chat message normalisation helpers shared by every provider."""

from typing import Optional

# Fields that should never be retained from Claude CLI auth-capture templates.
_CLAUDE_TEMPLATE_STRIPPED_FIELDS = {"tools", "thinking", "context_management", "tool_choice", "system"}


def _sanitize_claude_template(parsed: dict) -> dict:
    """Return a minimal Claude request template without prompt/tool baggage."""
    sanitized = dict(parsed)
    for field in _CLAUDE_TEMPLATE_STRIPPED_FIELDS:
        sanitized.pop(field, None)
    return sanitized


def _extract_oai_content(message: dict) -> Optional[str]:
    """Extract usable text from an OpenAI-format message, handling reasoning models.

    Reasoning models (GLM, Kimi, MiMo) may return content=null/empty while the actual
    output sits in 'reasoning' or 'reasoning_content'.  When content is empty and
    reasoning is present, we return None so the caller can handle it (e.g. retry
    without max_tokens).  We do NOT return reasoning as content because it's the
    model's internal chain-of-thought, not the answer.
    """
    return message.get("content") or None


def _is_reasoning_truncated(data: dict) -> bool:
    """Check if an OpenAI-format response was truncated mid-reasoning.

    Returns True when finish_reason is 'length' and content is empty but
    reasoning_content is present — meaning max_tokens was exhausted by the
    model's chain-of-thought before it could produce actual output.
    """
    choices = data.get("choices", [])
    if not choices:
        return False
    choice = choices[0]
    if choice.get("finish_reason") != "length":
        return False
    msg = choice.get("message", {})
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    return (not content) and bool(reasoning)


def _has_reasoning_without_content(data: dict) -> bool:
    """Return True when a choice carries reasoning fields but no user-visible content."""
    choices = data.get("choices", [])
    if not choices:
        return False
    msg = choices[0].get("message", {})
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    return (not content) and bool(reasoning)


def _strip_vision_content(content):
    """Collapse multimodal content blocks to text-only.

    If *content* is a list of dicts (OpenAI vision format), extract only the
    text parts and return a plain string.  Plain strings pass through unchanged.
    """
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return " ".join(p for p in parts if p) or ""
    return content


def build_oai_messages(system_prompt: Optional[str], messages: list, *, strip_vision: bool = False) -> list:
    """Build an OpenAI-format message list, prepending system prompt if present."""
    oai = []
    if system_prompt:
        oai.append({"role": "system", "content": system_prompt})
    for m in messages:
        content = m["content"]
        if strip_vision:
            content = _strip_vision_content(content)
        oai.append({"role": m["role"], "content": content})
    return oai


def _append_merged_message(messages: list[dict], role: str, content) -> None:
    """Append a message, coalescing adjacent messages from the same role.

    Content can be a string or a list (for vision/multimodal messages).
    """
    # Convert Pydantic models to native Python types
    if hasattr(content, "__iter__") and not isinstance(content, str):
        # List of Pydantic models - convert to dicts
        content = [
            c.model_dump() if hasattr(c, "model_dump") else
            c.dict() if hasattr(c, "dict") else c
            for c in content
        ]

    if messages and messages[-1]["role"] == role:
        prev_content = messages[-1]["content"]
        if isinstance(prev_content, str) and isinstance(content, str):
            messages[-1]["content"] += "\n\n" + content
        elif isinstance(prev_content, list) and isinstance(content, list):
            messages[-1]["content"].extend(content)
        else:
            # Mixed types, can't coalesce - append new message
            messages.append({"role": role, "content": content})
    else:
        messages.append({"role": role, "content": content})


def _normalize_chat_messages(messages: list) -> tuple[Optional[str], list[dict], list[dict]]:
    """Build provider-native and OpenAI-compatible views of the incoming chat history."""
    system_parts: list[str] = []
    merged_messages: list[dict] = []
    oai_messages: list[dict] = []

    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]

        # Convert content to native Python types
        if hasattr(content, "__iter__") and not isinstance(content, str):
            content = [
                c.model_dump() if hasattr(c, "model_dump") else
                c.dict() if hasattr(c, "dict") else c
                for c in content
            ]

        if role == "system":
            # System messages are always strings
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if c.get("text"))
            system_parts.append(content)
            _append_merged_message(oai_messages, role, content)
        elif role in ("user", "assistant"):
            _append_merged_message(merged_messages, role, content)
            _append_merged_message(oai_messages, role, content)

    if merged_messages and merged_messages[0]["role"] != "user":
        merged_messages.insert(0, {"role": "user", "content": "Continue."})

    return "\n\n".join(system_parts) or None, merged_messages, oai_messages
