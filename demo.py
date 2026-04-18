#!/usr/bin/env python3
"""Gradio chatbot demo for Pocket-Agent."""
import os, json, gradio as gr

# Reuse inference module
from inference import run as model_run, _load_model, SYSTEM_PROMPT

def chat(user_message, history):
    """Gradio chat handler with multi-turn support."""
    # Convert Gradio history format to our format
    model_history = []
    for turn in history:
        if isinstance(turn, (list, tuple)):
            # Old format: [(user, bot), ...]
            model_history.append({"role": "user", "content": turn[0]})
            if turn[1]:
                model_history.append({"role": "assistant", "content": turn[1]})
        elif isinstance(turn, dict):
            model_history.append(turn)

    # Run inference
    response = model_run(user_message, model_history)

    # Format for display
    if "<tool_call>" in response:
        # Parse and pretty-print the tool call
        try:
            tc_start = response.index("<tool_call>") + len("<tool_call>")
            tc_end = response.index("</tool_call>")
            tc_json = json.loads(response[tc_start:tc_end])
            display = f"🔧 **Tool Call**\n```json\n{json.dumps(tc_json, indent=2)}\n```"
        except (ValueError, json.JSONDecodeError):
            display = f"🔧 {response}"
    else:
        display = f"💬 {response}"

    return display

def main():
    # Pre-load model
    print("Loading model...")
    _load_model()
    print("Model loaded! Starting demo...")

    demo = gr.ChatInterface(
        fn=chat,
        title="🤖 Pocket-Agent",
        description=(
            "On-device mobile assistant with tool-calling capabilities. "
            "Supports: weather, calendar, unit conversion, currency conversion, SQL queries. "
            "Try asking things like 'What's the weather in Tokyo?' or 'Convert 100 USD to EUR'."
        ),
        examples=[
            "What's the weather in Paris?",
            "Convert 100 kilometers to miles",
            "Convert 500 USD to EUR",
            "Show my calendar for 2025-04-20",
            "Schedule Team Sync on 2025-05-01",
            "Show all users with age > 30",
            "Tell me a joke",
        ],
        theme=gr.themes.Soft(),
        retry_btn=None,
        undo_btn=None,
    )
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
