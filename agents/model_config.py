"""
agents/model_config.py
----------------------
Single source of truth for model selection and ChatOllama configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

# TEST profile: 50-case demo run
TEST = {
    "model": "gemma4:e2b",
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "num_ctx": 8192,
    "num_predict": 2048,
    "timeout": 180,
    "consensus_confidence": 72,
    "sample_size": 10,
    "sample_mode": "demo",
    "seed": 7,
}

# FULL profile: 500-case demo run
FULL = {
    "model": "gemma4:e2b",
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "num_ctx": 8192,
    "num_predict": 2048,
    "timeout": 180,
    "consensus_confidence": 72,
    "sample_size": 500,
    "sample_mode": "demo",
    "seed": 7,
}

# Change this single variable to switch profiles
ACTIVE_PROFILE = FULL

def make_llm() -> "ChatOllama":
    """
    Constructs and returns a ChatOllama instance using the ACTIVE_PROFILE.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=ACTIVE_PROFILE["model"],
        temperature=ACTIVE_PROFILE["temperature"],
        top_p=ACTIVE_PROFILE["top_p"],
        top_k=ACTIVE_PROFILE["top_k"],
        num_ctx=ACTIVE_PROFILE["num_ctx"],
        num_predict=ACTIVE_PROFILE["num_predict"],
        timeout=ACTIVE_PROFILE["timeout"],
    )
