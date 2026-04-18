"""Eval harness contract — the grader calls inference.run() with this interface."""
import json
from inference import run

def grade_example(prompt: str, history: list[dict], expected: str) -> dict:
    """Grade a single example. Returns dict with score and prediction."""
    prediction = run(prompt, history)
    return {
        "prompt": prompt,
        "expected": expected,
        "prediction": prediction,
    }
