"""
agents/model_config.py
----------------------
Single source of truth for model selection and ChatOllama configuration.
"""

from langchain_ollama import ChatOllama

# LOCAL_TRIAL profile: gemma4:e2b
LOCAL_TRIAL = {
    "model": "gemma4:e2b",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "num_ctx": 8192,
    "num_predict": 2048,
    "timeout": 180,
    "consensus_confidence": 72,
}

# FINAL profile: gemma4:31b
FINAL = {
    "model": "gemma4:31b",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "num_ctx": 16384,
    "num_predict": 4096,
    "timeout": 300,
    "consensus_confidence": 90,
}

# Change this single variable to switch profiles
ACTIVE_PROFILE = LOCAL_TRIAL

def make_llm() -> ChatOllama:
    """
    Constructs and returns a ChatOllama instance using the ACTIVE_PROFILE.
    """
    return ChatOllama(
        model=ACTIVE_PROFILE["model"],
        temperature=ACTIVE_PROFILE["temperature"],
        top_p=ACTIVE_PROFILE["top_p"],
        top_k=ACTIVE_PROFILE["top_k"],
        num_ctx=ACTIVE_PROFILE["num_ctx"],
        num_predict=ACTIVE_PROFILE["num_predict"],
        timeout=ACTIVE_PROFILE["timeout"],
    )
