#!/usr/bin/env python3
"""
Inference module for Pocket-Agent grader.
Exposes: def run(prompt: str, history: list[dict]) -> str

Loads a quantized GGUF model via llama-cpp-python for fast CPU inference.
No network imports. Fully offline.
"""
import os, json
from pathlib import Path

# ── Lazy-loaded globals ─────────────────────────────────────────────────
_model = None
_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "output", "gguf", "model-Q4_K_M.gguf")
)

SYSTEM_PROMPT = """You are a mobile assistant with these tools:

1. weather — location (string), unit ("C" or "F")
2. calendar — action ("list" or "create"), date ("YYYY-MM-DD"), title (string, optional for list)
3. convert — value (number), from_unit (string), to_unit (string)
4. currency — amount (number), from (ISO 4217 3-letter code), to (ISO 4217 3-letter code)
5. sql — query (string)

If a request matches a tool, reply ONLY with:
<tool_call>{"tool": "name", "args": {...}}</tool_call>

If no tool fits (chitchat, unsupported request, ambiguous with no history), reply with plain text. Never invent tools."""


def _load_model():
    """Load the quantized GGUF model lazily."""
    global _model
    if _model is not None:
        return _model

    from llama_cpp import Llama

    _model = Llama(
        model_path=_MODEL_PATH,
        n_ctx=1024,
        n_threads=4,
        n_batch=512,
        verbose=False,
    )
    return _model


def _build_prompt(prompt: str, history: list[dict]) -> str:
    """Build ChatML-formatted prompt string."""
    parts = []
    parts.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")

    # Add history
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    # Add current prompt
    parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
    parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


def run(prompt: str, history: list[dict]) -> str:
    """
    Run inference on the pocket-agent model.

    Args:
        prompt: The user's current message.
        history: List of prior turns, each a dict with 'role' and 'content'.

    Returns:
        Model response string (tool call or plain text refusal).
    """
    model = _load_model()
    full_prompt = _build_prompt(prompt, history)

    output = model(
        full_prompt,
        max_tokens=256,
        stop=["<|im_end|>", "<|im_start|>"],
        temperature=0.1,
        top_p=0.9,
        echo=False,
    )

    response = output["choices"][0]["text"].strip()
    return response


# ── CLI for quick testing ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        user_prompt = "What's the weather in Tokyo?"

    print(f"Prompt: {user_prompt}")
    result = run(user_prompt, [])
    print(f"Response: {result}")
