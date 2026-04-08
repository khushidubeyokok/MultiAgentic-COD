"""
agents/utils.py
---------------
Shared utilities for parsing LLM responses, especially for reasoning models (R1/OpenThinker).
"""

import json
import re

def strip_thoughts(text: str) -> str:
    """
    Remove <thought>...</thought> blocks from the model output.
    """
    if not text:
        return ""
    # Remove thought blocks
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Also handle cases where the thought block isn't closed (common on timeouts)
    cleaned = re.sub(r"<thought>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def parse_best_json(raw: str) -> dict:
    """
    Grabs the best-looking JSON block from the text.
    Steps:
      1. Strip <thought> blocks.
      2. Find all { ... } blocks.
      3. Try to parse the LARGEST one (usually the main result).
    """
    text = strip_thoughts(raw)
    
    # Find all blocks between { and }
    # Using a non-greedy .*? inside a lookahead/lookbehind is tricky for nested JSON,
    # so we'll use a more heuristic approach: find all { and find their matching }.
    # For speed and simplicity in LLM responses, we'll find the first '{' and last '}'
    # as the most likely candidate for the intended JSON object.
    
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return {}

    candidate = match.group(1)
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # If it failed, maybe there was preamble or postamble inside the braces?
        # Try to fix common trailing/leading comma issues or backticks
        candidate = re.sub(r"```json\s*", "", candidate)
        candidate = re.sub(r"\s*```", "", candidate)
        try:
            return json.loads(candidate)
        except:
            return {}
