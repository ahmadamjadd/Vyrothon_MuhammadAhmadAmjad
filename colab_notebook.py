# ============================================================
# 🤖 POCKET-AGENT — COLAB TRAINING NOTEBOOK
# ============================================================
# Copy-paste each cell into Google Colab (T4 GPU runtime)
# Total time: ~30-40 minutes
# ============================================================

# ============ CELL 1: Setup GPU Check ============
# Run this first to verify you have a T4 GPU
"""
!nvidia-smi
"""

# ============ CELL 2: Clone Your Repo ============
# First push your code to GitHub, then clone it here
# OR upload the files directly
"""
# Option A: Clone from GitHub (after you push)
!git clone https://github.com/YOUR_USERNAME/pocket-agent.git
%cd pocket-agent

# Option B: If you uploaded a zip
# !unzip pocket-agent.zip
# %cd pocket-agent
"""

# ============ CELL 3: Install Dependencies ============
"""
!pip install -q torch transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf
!pip install -q gradio
# Install llama-cpp-python (CPU version for inference)
!pip install -q llama-cpp-python
"""

# ============ CELL 4: Generate Training Data ============
"""
!python generate_data.py
"""
# Expected output: "Generated 1700 training examples → data/train.jsonl"

# ============ CELL 5: Fine-Tune the Model (~20 min) ============
"""
!python train.py
"""
# This will:
# - Download Qwen2.5-0.5B-Instruct (~1GB)
# - Load in 4-bit with QLoRA
# - Train for 3 epochs on 1700 examples
# - Save LoRA adapter to output/lora-adapter/

# ============ CELL 6: Merge LoRA + Quantize to GGUF ============
"""
!python merge_and_quantize.py
"""
# This will:
# - Merge LoRA adapter into base model
# - Clone llama.cpp for conversion tools
# - Convert to GGUF Q4_K_M format
# - Final model saved to output/gguf/model-Q4_K_M.gguf

# ============ CELL 7: Check Model Size ============
"""
import os
model_path = "output/gguf/model-Q4_K_M.gguf"
size_mb = os.path.getsize(model_path) / (1024*1024)
print(f"Model size: {size_mb:.1f} MB")
print(f"500MB gate: {'✅ PASS' if size_mb <= 500 else '❌ FAIL'}")
print(f"250MB bonus: {'✅ PASS' if size_mb <= 250 else '❌ NO'}")
"""

# ============ CELL 8: Evaluate ============
"""
!python evaluate.py starter/teacher_examples.jsonl
"""

# ============ CELL 9: Quick Test ============
"""
from inference import run

# Test tool calls
print("1:", run("What's the weather in Tokyo?", []))
print("2:", run("Convert 100 USD to EUR", []))
print("3:", run("Convert 50 kg to pounds", []))
print("4:", run("Show my calendar for 2025-04-20", []))
print("5:", run("Show all users with age > 30", []))

# Test refusal
print("6:", run("Tell me a joke", []))

# Test multi-turn
history = [
    {"role": "user", "content": "Convert 100 USD to GBP"},
    {"role": "assistant", "content": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "GBP"}}</tool_call>'}
]
print("7:", run("Now convert that to euros", history))
"""

# ============ CELL 10: Launch Gradio Demo ============
"""
!python demo.py
"""
# Click the public URL that appears to open the chatbot

# ============ CELL 11 (OPTIONAL): Quantize smaller for 250MB bonus ============
"""
# !QUANT_TYPE=Q2_K python merge_and_quantize.py
# import os
# size = os.path.getsize("output/gguf/model-Q2_K.gguf") / (1024*1024)
# print(f"Q2_K size: {size:.1f} MB — {'✅ BONUS' if size <= 250 else '❌'}")
"""
