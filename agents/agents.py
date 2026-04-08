"""
agents/agents.py
----------------
Defines the three specialist agent nodes (nodes in the LangGraph).
"""

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState

_LLM = ChatOllama(
    model="openthinker:7b",
    temperature=0.0,
    num_ctx=8192,
)

# Canonical list for validation
PHMRC = "Drowning, Poisonings, Other Cardiovascular Diseases, AIDS, Violent Death, Malaria, Other Cancers, Measles, Meningitis, Encephalitis, Diarrhea/Dysentery, Other Defined Causes of Child Deaths, Other Infectious Diseases, Hemorrhagic fever, Other Digestive Diseases, Bite of Venomous Animal, Fires, Falls, Sepsis, Pneumonia, Road Traffic"

def _parse_llm_json(raw: str, agent_name: str) -> dict:
    """Robustly parse JSON, stripping <thought> blocks if present."""
    # Strip <thought>...</thought>
    text = re.sub(r"<thought>.*?</thought>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Try to find JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
            
    # Fallback to parsing the raw text directly if no braces found (unlikely for openthinker)
    try:
        return json.loads(text)
    except:
        pass

    print(f"[WARN] {agent_name}: JSON parse failed. Returning error dict.")
    return {
        "agent_name": agent_name,
        "diagnosis": "Unknown",
        "confidence": "Low",
        "primary_reasoning": "Reasoning model output parse failed.",
        "error": True,
        "raw_response": raw
    }

def _call_llm(dossier: str, persona: str) -> dict:
    prompt = (
        f"You are a {persona}. Analyze the patient dossier and provide a diagnosis.\n\n"
        f"### PATIENT DOSSIER ###\n{dossier}\n\n"
        "### TASK ###\n"
        f"1. Reason through the clinical evidence.\n"
        f"2. Map the diagnosis to EXACTLY one of: {PHMRC}.\n"
        "3. Respond ONLY with a JSON object in this format:\n"
        "{\"diagnosis\": \"exact category from list\", \"confidence\": \"High/Medium/Low\", \"primary_reasoning\": \"2-3 sentence summary\"}"
    )
    
    response = _LLM.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    return _parse_llm_json(raw_text, persona)

def agent1_node(state: VAState) -> dict:
    return {"agent1_output": _call_llm(state["full_dossier"], "Pediatric Infectious Disease Specialist")}

def agent2_node(state: VAState) -> dict:
    return {"agent2_output": _call_llm(state["full_dossier"], "Pediatric Intensivist")}

def agent3_node(state: VAState) -> dict:
    return {"agent3_output": _call_llm(state["full_dossier"], "Pediatric Trauma and Nutritional Specialist")}
