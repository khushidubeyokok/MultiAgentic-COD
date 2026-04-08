"""
agents/agents.py
----------------
Defines the three specialist agent nodes (nodes in the LangGraph).
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, PHMRC_CATEGORIES

# Specialists have a slight temperature to encourage variety in reasoning.
_LLM_DIVERSE = ChatOllama(
    model="openthinker:7b",
    temperature=0.3,
    num_ctx=8192,
)

PHMRC_LIST = ", ".join(PHMRC_CATEGORIES)

def _call_llm(dossier: str, persona: str, focus_instructions: str) -> dict:
    prompt = (
        f"You are a {get_persona_title(persona)}. {focus_instructions}\n\n"
        f"### PATIENT DOSSIER ###\n{dossier}\n\n"
        "### TASK ###\n"
        "1. Reason through the clinical evidence using your specialized lens.\n"
        f"2. Map the diagnosis to EXACTLY one of: {PHMRC_LIST}.\n"
        "3. Provide your output in JSON format:\n"
        "{\"diagnosis\": \"exact category\", \"confidence\": \"High/Medium/Low\", \"primary_reasoning\": \"reason\"}\n\n"
        "4. IMPORTANT: After the JSON, repeat the category exactly once inside these tags: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]"
    )
    
    response = _LLM_DIVERSE.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    
    parsed = parse_best_json(raw_text)
    if not parsed:
        print(f"[WARN] {persona}: JSON parse failed. Returning error dict.")
        return {
            "agent_name": persona,
            "diagnosis": "Unknown",
            "confidence": "Low",
            "primary_reasoning": "Reasoning model output parse failed.",
            "error": True,
            "raw_response": raw_text
        }
    
    parsed["agent_name"] = persona
    return parsed

def get_persona_title(key: str) -> str:
    titles = {
        "specialist_id": "Pediatric Infectious Disease Specialist",
        "specialist_cc": "Pediatric Intensivist (Critical Care)",
        "specialist_tn": "Pediatric Trauma and Nutritional Specialist"
    }
    return titles.get(key, "Medical Consultant")

def agent1_node(state: VAState) -> dict:
    focus = "Focus on infectious triggers (fever, travel, rashes)."
    return {"agent1_output": _call_llm(state["full_dossier"], "specialist_id", focus)}

def agent2_node(state: VAState) -> dict:
    focus = "Focus on acute physiological failure (respiratory, neurological, organ failure)."
    return {"agent2_output": _call_llm(state["full_dossier"], "specialist_cc", focus)}

def agent3_node(state: VAState) -> dict:
    focus = "Focus on physical injuries, environmental factors, and nutritional status."
    return {"agent3_output": _call_llm(state["full_dossier"], "specialist_tn", focus)}
