#!/usr/bin/env python3
"""Fine-tune Qwen2.5-0.5B with QLoRA for tool calling."""
import os, json, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH  = os.environ.get("DATA_PATH",  "data/train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",  "output/lora-adapter")
EPOCHS     = int(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACC   = int(os.environ.get("GRAD_ACC", "4"))
LR         = float(os.environ.get("LR", "2e-4"))
MAX_SEQ_LEN= int(os.environ.get("MAX_SEQ_LEN", "768"))
LORA_R     = int(os.environ.get("LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "128"))

def load_data(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def format_chat(example, tokenizer):
    """Apply the tokenizer's chat template to get the formatted text."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

def main():
    print(f"🚀 Fine-tuning {BASE_MODEL}")
    print(f"   Data: {DATA_PATH}")
    print(f"   Output: {OUTPUT_DIR}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    raw_data = load_data(DATA_PATH)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(lambda ex: format_chat(ex, tokenizer), remove_columns=["messages"])
    dataset = dataset.shuffle(seed=42)
    print(f"   Training examples: {len(dataset)}")

    # Quantization config for QLoRA
    use_4bit = torch.cuda.is_available()
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("🏋️ Starting training...")
    trainer.train()

    # Save adapter
    print(f"💾 Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
