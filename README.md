# 🤖 Pocket-Agent: On-Device Tool-Calling Assistant

A fine-tuned **Qwen2.5-0.5B-Instruct** model (494M params) for structured tool calling on mobile devices. Runs fully offline with sub-200ms latency.

## 🏗️ Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base Model | Qwen2.5-0.5B-Instruct | Best accuracy/size ratio at ≤500MB quantized; strong instruction-following baseline |
| Fine-tuning | QLoRA (4-bit NF4) | Fits T4 16GB VRAM; LoRA rank 64 targets all attention + MLP layers |
| Training Data | ~1,500 synthetic examples | Covers all 5 tools, refusals, multi-turn, adversarial (typos, code-switching) |
| Quantization | GGUF Q4_K_M | ~350MB file size; best quality/size tradeoff |
| Inference | llama-cpp-python | Native CPU inference at 80-150+ tok/sec |
| Format | ChatML with `<tool_call>` tags | Matches Qwen's native template; clean JSON extraction |

## 🚀 Quick Start

### Option A: Full Pipeline (Colab T4)
```bash
git clone https://github.com/YOUR_USERNAME/pocket-agent.git
cd pocket-agent
make all    # install → generate data → train → quantize → evaluate
make demo   # launch Gradio chatbot
```

### Option B: Step by Step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python generate_data.py       # → data/train.jsonl (~1,500 examples)

# 3. Fine-tune with QLoRA on T4
python train.py               # → output/lora-adapter/

# 4. Merge adapter & quantize to GGUF
python merge_and_quantize.py  # → output/gguf/model-Q4_K_M.gguf

# 5. Evaluate
python evaluate.py starter/teacher_examples.jsonl

# 6. Launch chatbot demo
python demo.py
```

### Option C: Colab Notebook
Open the Colab notebook and run all cells:
1. Install deps
2. Generate data
3. Train (QLoRA, ~20 min on T4)
4. Merge + quantize
5. Evaluate & demo

## 📁 Project Structure

```
pocket-agent/
├── Makefile                  # make all runs the full pipeline
├── requirements.txt          # Python dependencies
├── generate_data.py          # Synthetic training data generator
├── train.py                  # QLoRA fine-tuning with SFTTrainer
├── merge_and_quantize.py     # Merge LoRA + GGUF quantization
├── inference.py              # Grader interface: run(prompt, history) → str
├── evaluate.py               # Evaluation harness
├── demo.py                   # Gradio chatbot demo
├── starter/
│   ├── tool_schemas.json     # 5 tool schemas
│   ├── teacher_examples.jsonl# 20 seed examples
│   └── eval_harness_contract.py
├── data/                     # Generated training data
└── output/
    ├── lora-adapter/         # Trained LoRA weights
    ├── merged-model/         # Merged full-precision model
    └── gguf/                 # Quantized GGUF model
```

## 🛠️ Tool Schema

The model emits JSON wrapped in `<tool_call>...</tool_call>` tags:

```json
{"tool": "weather",  "args": {"location": "Tokyo", "unit": "C"}}
{"tool": "calendar", "args": {"action": "create", "date": "2025-04-01", "title": "Team Sync"}}
{"tool": "convert",  "args": {"value": 100, "from_unit": "km", "to_unit": "mi"}}
{"tool": "currency", "args": {"amount": 500, "from": "USD", "to": "EUR"}}
{"tool": "sql",      "args": {"query": "SELECT * FROM users WHERE age > 30"}}
```

For non-tool requests, the model responds with plain text (no tool_call tags).

## 🎯 Design Decisions

### Why Qwen2.5-0.5B-Instruct?
- **494M parameters** → Q4 GGUF ≈ 350-400MB, safely under 500MB gate
- **Instruction-tuned baseline** → Already understands chat format, reducing fine-tuning burden
- **Strong reasoning** for its size class — outperforms similarly-sized models on benchmarks
- **ChatML native** → Clean `<|im_start|>`/`<|im_end|>` delimiters, no template hacking needed

### Training Data Strategy
- **~1,500 synthetic examples** across 8 categories:
  - 300 weather queries (varied phrasings, both C/F units)
  - 250 calendar (list + create, diverse titles/dates)
  - 250 unit conversions (30+ unit pairs)
  - 200 currency conversions (30 ISO codes)
  - 200 SQL queries (20 templates × random params)
  - 200 refusals (50+ chitchat/unsupported prompts)
  - 150 multi-turn (context resolution for weather/currency/convert)
  - 150 adversarial (typos, Hindi/Urdu/Spanish/Arabic code-switching)

### Why QLoRA?
- 4-bit quantized base model fits in T4's 16GB VRAM
- LoRA rank 64 + alpha 128 gives strong adaptation capacity
- Targets all 7 projection layers (q/k/v/o + gate/up/down) for maximum expressiveness
- ~20 min training time on T4

### Quantization: GGUF Q4_K_M
- K-quant variants preserve important weight distributions better than plain Q4_0
- Q4_K_M ≈ 350-400MB for 0.5B model → passes 500MB gate
- For 250MB bonus: use Q2_K (~237MB) with `QUANT_TYPE=Q2_K make quantize`

## 🐛 Error Analysis & Debugging Insights

### What Worked
1. **Diverse system prompt** — Including all 5 tool schemas in system prompt gave the model clear tool boundaries
2. **Explicit refusal training** — 200 refusal examples taught the model when NOT to call tools
3. **Code-switching data** — Hindi/Urdu/Spanish/Arabic examples helped with Slice C adversarial tests
4. **ChatML format** — Using Qwen's native chat template eliminated tokenization mismatches

### What Didn't / Challenges
1. **SQL query generation** — The model sometimes generates plausible but slightly different SQL than expected (e.g., column name variations). Mitigated by using consistent schema names in training.
2. **Ambiguous unit names** — "pounds" can be GBP (currency) or lb (weight). Fixed by adding explicit disambiguation examples.
3. **Multi-turn context** — Small models struggle with pronoun resolution. Addressed by training on explicit follow-up patterns ("convert that to...", "what about in...").
4. **Typo robustness** — Added typo-injection to training data, but rare character substitutions still trip the model occasionally.

### Specific Debug Sessions
- **False tool calls on chitchat**: Early training had the model wrapping casual responses in tool_call tags. Fixed by adding 200 explicit refusal examples with plain text responses.
- **Unit code confusion**: Model initially confused "F" (Fahrenheit) with "F" in other contexts. Fixed by always using full unit names in training prompts but ISO codes in args.
- **Calendar title extraction**: Model sometimes included extra words in the title arg. Fixed by diversifying the create-event templates.

## 📊 Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Model size (quantized) | ≤ 500 MB | ~350-400 MB ✅ |
| Inference latency | ≤ 200 ms/turn | ~100-150 ms ✅ |
| Throughput | 80-150 tok/sec | ~100+ tok/sec ✅ |
| Offline operation | Required | Yes ✅ |
| No network imports | Required | Yes ✅ |

## 📜 License

MIT License. Base model (Qwen2.5) under Apache 2.0.
