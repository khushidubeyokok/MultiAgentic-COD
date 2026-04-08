"""
agents/adjudicator.py
---------------------
Defines the adjudicator_node LangGraph node function.
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

# ── PHMRC category list ──────────────────────────────────────────────────────
PHMRC_CATEGORIES = [
    "Drowning", "Poisonings", "Other Cardiovascular Diseases", "AIDS",
    "Violent Death", "Malaria", "Other Cancers", "Measles", "Meningitis",
    "Encephalitis", "Diarrhea/Dysentery", "Other Defined Causes of Child Deaths",
    "Other Infectious Diseases", "Hemorrhagic fever", "Other Digestive Diseases",
    "Bite of Venomous Animal", "Fires", "Falls", "Sepsis", "Pneumonia",
    "Road Traffic",
]

def _strip_thoughts(text: str) -> str:
    """Strip <thought> blocks."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def _parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON, stripping <thought> blocks."""
    text = _strip_thoughts(raw)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {
        "final_diagnosis": "Parse Error",
        "mapped_category": "Other Defined Causes of Child Deaths",
        "confidence_score": 0,
        "final_reasoning": "Adjudicator output parse failed.",
        "winning_agent": "None",
        "error": True,
        "raw_response": raw,
    }

def _build_user_prompt(state: VAState) -> str:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]
    def _get(d: dict, k: str) -> str: return str(d.get(k, "Unknown"))
    
    return f"""### PATIENT DOSSIER ###
{state["full_dossier"][:3000]}

### SPECIALIST INPUTS ###
Agent 1: {_get(a1, "diagnosis")} | Reasoning: {_get(a1, "primary_reasoning")}
Agent 2: {_get(a2, "diagnosis")} | Reasoning: {_get(a2, "primary_reasoning")}
Agent 3: {_get(a3, "diagnosis")} | Reasoning: {_get(a3, "primary_reasoning")}

### CRITIC REPORT ###
{state.get("critique", "No critique available.")}

### TASK ###
Render a final adjudication in JSON format.
JSON Schema: {{"final_diagnosis": "text", "mapped_category": "EXACT_CATEGORY", "confidence_score": 0-100, "final_reasoning": "text", "winning_agent": "name"}}
Categories: {", ".join(PHMRC_CATEGORIES)}
Respond ONLY with the JSON block."""

def adjudicator_node(state: VAState) -> dict:
    prompt = _build_user_prompt(state)
    response = _LLM.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    
    result = _parse_llm_json(raw_text)
    
    # Simple remap (can be improved if needed)
    cat = result.get("mapped_category", "Other Defined Causes of Child Deaths")
    if cat not in PHMRC_CATEGORIES:
        # Try substring search
        for c in PHMRC_CATEGORIES:
            if c.lower() in str(cat).lower():
                cat = c
                break
        else:
            cat = "Other Defined Causes of Child Deaths"
            
    return {
        "final_diagnosis":  str(result.get("final_diagnosis", cat)),
        "mapped_category":  str(cat),
        "confidence_score": int(result.get("confidence_score", 0)),
        "final_reasoning":  str(result.get("final_reasoning", "No reasoning.")),
    }
