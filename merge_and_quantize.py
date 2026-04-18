#!/usr/bin/env python3
"""Merge LoRA adapter into base model and quantize to GGUF."""
import os, sys, subprocess, shutil
from pathlib import Path

BASE_MODEL   = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR  = os.environ.get("ADAPTER_DIR", "output/lora-adapter")
MERGED_DIR   = os.environ.get("MERGED_DIR", "output/merged-model")
GGUF_DIR     = os.environ.get("GGUF_DIR", "output/gguf")
QUANT_TYPE   = os.environ.get("QUANT_TYPE", "Q4_K_M")

def merge_adapter():
    """Merge LoRA adapter into the base model (full precision)."""
    print(f"🔀 Merging adapter from {ADAPTER_DIR} into {BASE_MODEL}...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"✅ Merged model saved to {MERGED_DIR}")

def convert_to_gguf():
    """Convert merged HF model to GGUF format using llama.cpp."""
    print("📦 Converting to GGUF...")
    os.makedirs(GGUF_DIR, exist_ok=True)
    fp16_path = os.path.join(GGUF_DIR, "model-fp16.gguf")
    quant_path = os.path.join(GGUF_DIR, f"model-{QUANT_TYPE}.gguf")

    # Install llama-cpp-python if needed (has conversion tools)
    try:
        import llama_cpp
        llama_cpp_dir = os.path.dirname(os.path.dirname(llama_cpp.__file__))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])

    # Try using the convert script from llama.cpp repo
    # First, clone llama.cpp if not present
    llama_cpp_repo = "llama.cpp"
    if not os.path.exists(llama_cpp_repo):
        print("📥 Cloning llama.cpp for conversion tools...")
        subprocess.check_call([
            "git", "clone", "--depth=1",
            "https://github.com/ggerganov/llama.cpp.git", llama_cpp_repo
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            os.path.join(llama_cpp_repo, "requirements.txt")
        ])

    # Convert HF to GGUF (FP16)
    convert_script = os.path.join(llama_cpp_repo, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Try alternative path
        convert_script = os.path.join(llama_cpp_repo, "convert-hf-to-gguf.py")

    print(f"  Converting HF → GGUF FP16...")
    subprocess.check_call([
        sys.executable, convert_script,
        MERGED_DIR,
        "--outfile", fp16_path,
        "--outtype", "f16",
    ])

    # Quantize
    print(f"  Quantizing to {QUANT_TYPE}...")
    # Try to find llama-quantize
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        # Build it
        print("  Building llama-quantize...")
        subprocess.check_call(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                            cwd=llama_cpp_repo)
        subprocess.check_call(["cmake", "--build", "build", "--target", "llama-quantize", "-j"],
                            cwd=llama_cpp_repo)
        quantize_bin = os.path.join(llama_cpp_repo, "build", "bin", "llama-quantize")

    subprocess.check_call([quantize_bin, fp16_path, quant_path, QUANT_TYPE])

    # Report size
    size_mb = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"✅ Quantized model: {quant_path} ({size_mb:.1f} MB)")

    # Cleanup fp16
    if os.path.exists(fp16_path):
        os.remove(fp16_path)
        print(f"  Removed intermediate FP16 file")

    return quant_path

def main():
    merge_adapter()
    quant_path = convert_to_gguf()
    print(f"\n🎉 Done! Quantized model at: {quant_path}")

    size_mb = os.path.getsize(quant_path) / (1024*1024)
    if size_mb <= 250:
        print(f"   🏆 BONUS: Model is {size_mb:.1f} MB (≤250 MB)")
    elif size_mb <= 500:
        print(f"   ✅ Model is {size_mb:.1f} MB (≤500 MB gate passed)")
    else:
        print(f"   ⚠️  Model is {size_mb:.1f} MB (exceeds 500 MB gate!)")

if __name__ == "__main__":
    main()
