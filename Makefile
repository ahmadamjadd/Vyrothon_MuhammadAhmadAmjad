# ── Pocket-Agent: On-Device Tool-Calling Assistant ──────────────────────
# Reproducible pipeline: make all

BASE_MODEL   ?= Qwen/Qwen2.5-0.5B-Instruct
ADAPTER_DIR  ?= output/lora-adapter
MERGED_DIR   ?= output/merged-model
GGUF_DIR     ?= output/gguf
QUANT_TYPE   ?= Q4_K_M
DATA_PATH    ?= data/train.jsonl

export BASE_MODEL ADAPTER_DIR MERGED_DIR GGUF_DIR QUANT_TYPE DATA_PATH

.PHONY: all install data train quantize eval demo clean

all: install data train quantize eval
	@echo "✅ Full pipeline complete!"

install:
	pip install -r requirements.txt

data:
	python generate_data.py

train:
	python train.py

quantize:
	python merge_and_quantize.py

eval:
	python evaluate.py starter/teacher_examples.jsonl

demo:
	python demo.py

clean:
	rm -rf output/ data/train.jsonl llama.cpp __pycache__
