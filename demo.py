#!/usr/bin/env python3
"""Enhanced Gradio chatbot demo for Pocket-Agent with rich UI."""
import os, json, time
import gradio as gr

from inference import run as model_run, _load_model

TOOL_EMOJIS = {
    "weather": "🌤️",
    "calendar": "📅",
    "convert": "🔄",
    "currency": "💱",
    "sql": "🗄️",
}

def format_response(response):
    """Pretty-format model response for display."""
    if "<tool_call>" in response:
        try:
            tc_start = response.index("<tool_call>") + len("<tool_call>")
            tc_end = response.index("</tool_call>")
            tc_json = json.loads(response[tc_start:tc_end])
            tool = tc_json.get("tool", "unknown")
            args = tc_json.get("args", {})
            emoji = TOOL_EMOJIS.get(tool, "🔧")

            # Build a nice formatted output
            args_lines = "\n".join(f"  • **{k}**: `{v}`" for k, v in args.items())
            display = f"{emoji} **Tool Call: `{tool}`**\n\n{args_lines}\n\n---\n*Raw:* `{response}`"
            return display
        except (ValueError, json.JSONDecodeError):
            return f"🔧 {response}"
    else:
        return f"💬 {response}"

def chat(user_message, history):
    """Handle chat with multi-turn support."""
    if not user_message.strip():
        return ""

    # Convert Gradio history to model format
    model_history = []
    for turn in history:
        if isinstance(turn, dict):
            model_history.append(turn)
        elif isinstance(turn, (list, tuple)):
            model_history.append({"role": "user", "content": turn[0]})
            if turn[1]:
                # Extract raw response if we formatted it
                raw = turn[1]
                if "*Raw:*" in raw:
                    try:
                        raw = raw.split("*Raw:* `")[1].rstrip("`")
                    except IndexError:
                        pass
                model_history.append({"role": "assistant", "content": raw})

    # Run inference with timing
    t0 = time.time()
    response = model_run(user_message, model_history)
    latency = (time.time() - t0) * 1000

    # Format for display
    display = format_response(response)
    display += f"\n\n⚡ *{latency:.0f}ms*"
    return display

# ── Build UI ────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🤖 Pocket-Agent

**On-device mobile assistant** fine-tuned from Qwen2.5-0.5B-Instruct

### Available Tools
| Tool | Description | Example |
|------|------------|---------|
| 🌤️ weather | Get weather info | "Weather in Tokyo" |
| 📅 calendar | List/create events | "Schedule meeting on 2025-05-01" |
| 🔄 convert | Unit conversion | "Convert 100 km to miles" |
| 💱 currency | Currency exchange | "500 USD to EUR" |
| 🗄️ sql | Database queries | "Show all users with age > 30" |

*Unsupported requests get a polite refusal. Multi-turn context is supported!*
"""

EXAMPLES = [
    "What's the weather in Tokyo?",
    "Weather in Paris in Fahrenheit",
    "Convert 100 kilometers to miles",
    "How many pounds is 75 kilograms?",
    "Convert 500 USD to EUR",
    "How much is 1000 JPY in GBP?",
    "Show my calendar for 2025-04-20",
    "Schedule Team Sync on 2025-05-01",
    "Show all users with age greater than 30",
    "Get total sales by region",
    "Tell me a joke",
    "Send an email to John",
    "مجھے Tokyo کا weather بتاؤ",
    "Convierte 100 km a millas",
]

def main():
    print("⏳ Loading model...")
    _load_model()
    print("✅ Model loaded!")

    with gr.Blocks(
        title="Pocket-Agent",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
    ) as demo:
        gr.Markdown(DESCRIPTION)

        chatbot = gr.ChatInterface(
            fn=chat,
            examples=EXAMPLES,
            retry_btn="🔄 Retry",
            undo_btn="↩️ Undo",
            clear_btn="🗑️ Clear",
        )

    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
