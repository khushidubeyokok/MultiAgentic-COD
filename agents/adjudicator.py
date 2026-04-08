"""
agents/adjudicator.py
---------------------
Defines the adjudicator_node LangGraph node function.
"""

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, strip_thoughts

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

def _build_user_prompt(state: VAState) -> str:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]
    def _get(d: dict, k: str) -> str: return str(d.get(k, "Unknown"))
    
    return f"""### PATIENT DOSSIER (Partial) ###
{state["full_dossier"][:5000]}

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
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]
    
    # ── CONSENSUS FORCE ───────────────────────────────────────────────────────
    # If all three specialists agree on an EXACT PHMRC category, Bypassing LLM call.
    d1, d2, d3 = a1.get("diagnosis"), a2.get("diagnosis"), a3.get("diagnosis")
    if d1 == d2 == d3 and d1 in PHMRC_CATEGORIES:
        print(f"[INFO] Adjudicator: Unanimous consensus detected ({d1}). Applying Consensus Force.")
        return {
            "final_diagnosis":  str(d1),
            "mapped_category":  str(d1),
            "confidence_score": 100,
            "final_reasoning":  "Unanimous consensus among all three medical specialists.",
            "winning_agent":    "Consensus",
        }

    # Otherwise, perform LLM adjudication
    prompt = _build_user_prompt(state)
    response = _LLM.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    
    result = parse_best_json(raw_text)
    
    # Map the diagnosis back to canonical PHMRC if possible
    cat = result.get("mapped_category", result.get("final_diagnosis", "Other Defined Causes of Child Deaths"))
    
    # Simple remap
    if cat not in PHMRC_CATEGORIES:
        for c in PHMRC_CATEGORIES:
            if c.lower() in str(cat).lower():
                cat = c
                break
        else:
            cat = "Other Defined Causes of Child Deaths"
            
    return {
        "final_diagnosis":  str(result.get("final_diagnosis", cat)),
        "mapped_category":  str(cat),
        "confidence_score": int(result.get("confidence_score", 50)),
        "final_reasoning":  str(result.get("final_reasoning", "No reasoning.")),
        "winning_agent":    str(result.get("winning_agent", "Adjudicator Override")),
    }
