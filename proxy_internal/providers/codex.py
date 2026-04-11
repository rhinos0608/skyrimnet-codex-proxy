"""Codex CLI (OpenAI) provider — spawns the local Codex CLI subprocess."""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from typing import Optional

from fastapi import HTTPException

from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


def _get_codex_command() -> tuple[str, list[str]]:
    """
    Get the correct command to run Codex CLI on the current platform.

    On Windows, .cmd files need to be executed through cmd.exe.
    Returns (executable, prefix_args) where prefix_args are additional args
    that go before the codex arguments.
    """
    import proxy
    if not proxy.CODEX_PATH:
        return ("", [])

    # On Windows, .cmd/.bat files must be run through cmd.exe
    if sys.platform == "win32" and proxy.CODEX_PATH.lower().endswith((".cmd", ".bat")):
        # Use full path to cmd.exe to ensure it's found regardless of cwd
        cmd_exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
        return (cmd_exe, ["/c", proxy.CODEX_PATH])

    return (proxy.CODEX_PATH, [])


def _convert_messages_to_codex_input(messages: list) -> list:
    """Convert OpenAI-format messages to Codex Responses API input format."""
    input_items = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            # System messages go in instructions field, not input
            continue
        elif role == "user":
            input_items.append({"role": "user", "content": content})
        elif role == "assistant":
            input_items.append({"role": "assistant", "content": content})
    return input_items


def _create_isolated_codex_home() -> tuple[str, dict]:
    """
    Create an isolated HOME directory with minimal Codex config.
    Returns (isolated_home_path, env_dict).
    """
    import proxy
    # Create isolated home directory
    isolated_home = tempfile.mkdtemp(prefix="codex_isolated_")
    codex_dir = os.path.join(isolated_home, ".codex")
    os.makedirs(codex_dir, exist_ok=True)

    # Write minimal config.toml (no instructions, no skills)
    config_content = f'''# Minimal isolated Codex config
model = "{proxy.DEFAULT_CODEX_MODEL}"

[approval]
mode = "suggest"

[features]
# Disable all optional features for clean isolation
'''
    with open(os.path.join(codex_dir, "config.toml"), "w", encoding="utf-8") as f:
        f.write(config_content)

    # Copy auth.json from real home if it exists
    real_auth_path = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(real_auth_path):
        import shutil
        shutil.copy2(real_auth_path, os.path.join(codex_dir, "auth.json"))

    # Build isolated environment
    env = os.environ.copy()
    env["HOME"] = isolated_home
    env["USERPROFILE"] = isolated_home  # Windows
    env["CODEX_HOME"] = codex_dir

    return isolated_home, env


def _cleanup_isolated_home(isolated_home: str) -> None:
    """Clean up the isolated HOME directory after use."""
    import shutil
    try:
        shutil.rmtree(isolated_home, ignore_errors=True)
    except Exception:
        pass


def _build_codex_exec_args(model: str) -> tuple[str, ...]:
    """Return the Codex CLI argv with fast-mode overrides."""
    import proxy
    return (
        "exec",
        "--model", model,
        "-c", f'model_reasoning_effort="{proxy.CODEX_FAST_REASONING_EFFORT}"',
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-",
    )


async def call_codex_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Spawn Codex CLI subprocess with isolated HOME (clean config, no global instructions)."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.CODEX_PATH:
        raise HTTPException(status_code=503, detail="Codex CLI not installed")

    # Build prompt from messages
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m["role"].capitalize()
        prompt_parts.append(f"{role}: {m['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Codex CLI {model} ({len(messages)} msgs, isolated)")
    start = time.time()

    isolated_home = None
    try:
        # Create isolated HOME with clean config
        isolated_home, env = _create_isolated_codex_home()

        executable, prefix_args = _get_codex_command()
        logger.info(f"[{request_id}] Executing: {executable} {' '.join(prefix_args)} exec --model {model}")
        proc = await asyncio.create_subprocess_exec(
            executable,
            *prefix_args,
            *_build_codex_exec_args(model),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,  # Run from isolated directory
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=full_prompt.encode("utf-8")), timeout=180
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace")
            logger.error(f"[{request_id}] Codex CLI failed: {error_text[:300]}")
            raise HTTPException(status_code=500, detail=f"Codex CLI error: {error_text[:200]}")

        # Parse JSONL output and extract agent_message text
        response_text = ""
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        for line in raw_output.split("\n"):
            if line.strip():
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")

                    # Current Codex CLI format: {"id": ..., "msg": {"type": "agent_message", "message": "..."}}
                    msg = event.get("msg", {})
                    if isinstance(msg, dict) and msg.get("type") == "agent_message":
                        text = msg.get("message", "")
                        if text:
                            response_text = text

                    # Legacy format: {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
                    elif event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text = item.get("text", "")
                    elif event_type == "message.completed":
                        for content in event.get("message", {}).get("content", []):
                            if content.get("type") == "output_text":
                                response_text = content.get("text", "")
                except json.JSONDecodeError:
                    pass

        if not response_text:
            logger.warning(f"[{request_id}] Codex CLI: no agent_message found in output, raw length={len(raw_output)}")
            response_text = raw_output

        logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s)")
        return response_text
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Codex CLI timeout")
        raise HTTPException(status_code=504, detail="Codex CLI timeout")
    finally:
        if isolated_home:
            _cleanup_isolated_home(isolated_home)


async def call_codex_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Spawn Codex CLI subprocess with isolated HOME and yield output as SSE stream."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.CODEX_PATH:
        yield 'data: {"error": "Codex CLI not installed"}\n\n'
        yield "data: [DONE]\n\n"
        return

    # Build prompt from messages
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m["role"].capitalize()
        prompt_parts.append(f"{role}: {m['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Codex CLI {model} ({len(messages)} msgs, isolated stream)")
    start = time.time()

    # Send initial chunk with role
    role_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    isolated_home = None
    try:
        # Create isolated HOME with clean config
        isolated_home, env = _create_isolated_codex_home()

        executable, prefix_args = _get_codex_command()
        proc = await asyncio.create_subprocess_exec(
            executable,
            *prefix_args,
            *_build_codex_exec_args(model),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,
        )

        # Collect all output first (Codex CLI buffers output)
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=full_prompt.encode("utf-8")), timeout=180
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace")
            logger.error(f"[{request_id}] Codex CLI failed: {error_text[:300]}")
            err_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": f"[Codex error: {error_text[:100]}]"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Parse JSONL and extract agent_message
        response_text = ""
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        for line in raw_output.split("\n"):
            if line.strip():
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")

                    # Current format: {"id": ..., "msg": {"type": "agent_message", "message": "..."}}
                    msg = event.get("msg", {})
                    if isinstance(msg, dict) and msg.get("type") == "agent_message":
                        text = msg.get("message", "")
                        if text:
                            response_text = text

                    # Legacy format
                    elif event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text = item.get("text", "")
                    elif event_type == "message.completed":
                        for content in event.get("message", {}).get("content", []):
                            if content.get("type") == "output_text":
                                response_text = content.get("text", "")
                except json.JSONDecodeError:
                    pass

        if not response_text:
            logger.warning(f"[{request_id}] Codex CLI: no agent_message found in stream output, raw length={len(raw_output)}")
            response_text = raw_output

        # Stream the response in chunks for realistic streaming feel
        if response_text:
            chunk_size = 20  # characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk_text = response_text[i:i+chunk_size]
                oai_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(oai_chunk)}\n\n"

        # Final chunk
        stop_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s, simulated stream)")
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Codex CLI timeout")
        err, done = yield_sse_error(model, "[Codex CLI timeout]")
        yield err; yield done
    except Exception as e:
        logger.error(f"[{request_id}] Codex CLI error: {e}")
        err, done = yield_sse_error(model, f"[Codex CLI error: {e}]")
        yield err; yield done
    finally:
        if isolated_home:
            _cleanup_isolated_home(isolated_home)
