"""
agents/llm_config.py
--------------------
Centralised LLM configuration.
Supports 4 backends: Groq, NVIDIA NIM, Google Gemini, and Ollama (local).

Set LLM_BACKEND in .env to switch:
  - "groq"   → Groq cloud (Llama 3.3 70B, free 14K req/day)
  - "nvidia"  → NVIDIA NIM (Llama 3.1 405B, 1000 free credits)
  - "gemini"  → Google Gemini (2.5 Flash, free 1000 req/day)
  - "ollama"  → Local Ollama (any model you have pulled)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Backend selection ──────────────────────────────────────────────────────
BACKEND = os.getenv("LLM_BACKEND", "groq")

# ── Groq ───────────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── NVIDIA NIM ─────────────────────────────────────────────────────────────
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-405b-instruct")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")  # starts with nvapi-

# ── Google Gemini ──────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Ollama (local) ─────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_CTX = int(os.getenv("OLLAMA_CTX", "8192"))


def get_llm(temperature: float = 0.3):
    """Return a LangChain chat model based on the configured backend."""

    if BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )

    elif BACKEND == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=NVIDIA_MODEL,
            api_key=NVIDIA_API_KEY,
            temperature=temperature,
        )

    elif BACKEND == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    else:  # ollama (local)
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=temperature,
            num_ctx=OLLAMA_CTX,
        )
