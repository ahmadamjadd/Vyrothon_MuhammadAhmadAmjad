#!/usr/bin/env python3
"""Evaluate the model against a test set."""
import json, sys, os, time, re

from inference import run as model_run, _load_model


def parse_tool_call(text):
    """Extract tool call JSON from response text."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def is_tool_call(text):
    return "<tool_call>" in text and "</tool_call>" in text


def score_example(predicted, expected):
    """Score a single prediction against expected output.
    Returns (score, details_str)."""
    pred_tc = parse_tool_call(predicted)
    exp_tc = parse_tool_call(expected)

    pred_is_tc = pred_tc is not None
    exp_is_tc = exp_tc is not None

    # Case 1: Expected refusal
    if not exp_is_tc:
        if pred_is_tc:
            return -0.5, "❌ Emitted tool call when refusal expected"
        else:
            return 1.0, "✅ Correct refusal"

    # Case 2: Expected tool call
    if not pred_is_tc:
        return 0.0, "❌ No tool call when one was expected"

    # Check tool name
    if pred_tc.get("tool") != exp_tc.get("tool"):
        return 0.0, f"❌ Wrong tool: got {pred_tc.get('tool')}, expected {exp_tc.get('tool')}"

    # Check args
    pred_args = pred_tc.get("args", {})
    exp_args = exp_tc.get("args", {})

    all_correct = True
    details = []
    for key, exp_val in exp_args.items():
        pred_val = pred_args.get(key)
        if pred_val is None:
            all_correct = False
            details.append(f"missing {key}")
            continue

        # Numeric comparison with ±1% tolerance
        if isinstance(exp_val, (int, float)) and isinstance(pred_val, (int, float)):
            if exp_val == 0:
                if pred_val != 0:
                    all_correct = False
                    details.append(f"{key}: {pred_val} != {exp_val}")
            elif abs(pred_val - exp_val) / abs(exp_val) > 0.01:
                all_correct = False
                details.append(f"{key}: {pred_val} != {exp_val}")
        elif str(pred_val).lower() != str(exp_val).lower():
            all_correct = False
            details.append(f"{key}: '{pred_val}' != '{exp_val}'")

    if all_correct:
        return 1.0, f"✅ Exact match ({exp_tc['tool']})"
    else:
        return 0.5, f"⚠️  Correct tool, wrong args: {'; '.join(details)}"


def evaluate(test_path):
    """Run evaluation on a test file."""
    print(f"📊 Evaluating on {test_path}")
    print("=" * 60)

    # Pre-load model
    _load_model()

    with open(test_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    total_score = 0
    total_time = 0
    results = []

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        history = ex.get("history", [])
        expected = ex["expected"]

        t0 = time.time()
        predicted = model_run(prompt, history)
        elapsed = time.time() - t0
        total_time += elapsed

        score, detail = score_example(predicted, expected)
        total_score += score
        results.append({
            "idx": i,
            "prompt": prompt[:60],
            "score": score,
            "detail": detail,
            "latency_ms": elapsed * 1000,
        })

        print(f"  [{i+1:2d}] {score:+5.1f}  {elapsed*1000:6.0f}ms  {detail}")
        if score < 1.0:
            print(f"       Prompt:   {prompt[:80]}")
            print(f"       Expected: {expected[:80]}")
            print(f"       Got:      {predicted[:80]}")

    print("=" * 60)
    avg_latency = (total_time / len(examples)) * 1000
    print(f"Total score:    {total_score:.1f} / {len(examples)}")
    print(f"Accuracy:       {total_score/len(examples)*100:.1f}%")
    print(f"Avg latency:    {avg_latency:.0f} ms/turn")
    print(f"Total time:     {total_time:.1f}s")

    if avg_latency <= 200:
        print("✅ Latency gate: PASSED")
    else:
        print(f"⚠️  Latency gate: MARGINAL ({avg_latency:.0f}ms > 200ms target)")

    return total_score, len(examples)


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "starter/teacher_examples.jsonl"
    evaluate(test_file)
