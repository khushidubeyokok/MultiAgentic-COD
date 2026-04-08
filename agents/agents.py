"""
agents/agents.py
----------------
Defines the three specialist agent nodes (nodes in the LangGraph).
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json

# Specialists have a slight temperature to encourage variety in reasoning.
_LLM_DIVERSE = ChatOllama(
    model="openthinker:7b",
    temperature=0.3,
    num_ctx=8192,
)

# Canonical list for validation
PHMRC = "Drowning, Poisonings, Other Cardiovascular Diseases, AIDS, Violent Death, Malaria, Other Cancers, Measles, Meningitis, Encephalitis, Diarrhea/Dysentery, Other Defined Causes of Child Deaths, Other Infectious Diseases, Hemorrhagic fever, Other Digestive Diseases, Bite of Venomous Animal, Fires, Falls, Sepsis, Pneumonia, Road Traffic"

def _call_llm(dossier: str, persona: str, focus_instructions: str) -> dict:
    prompt = (
        f"You are a {get_persona_title(persona)}. {focus_instructions}\n\n"
        f"### PATIENT DOSSIER ###\n{dossier}\n\n"
        "### TASK ###\n"
        "1. Reason through the clinical evidence using your specialized lens.\n"
        f"2. Map the diagnosis to EXACTLY one of these 21 categories: {PHMRC}.\n"
        "3. Provide your output in JSON format:\n"
        "{\"diagnosis\": \"exact category from list\", \"confidence\": \"High/Medium/Low\", \"primary_reasoning\": \"2-3 sentence summary\"}"
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
    focus = (
        "Focus your analysis on infectious triggers. Look closely at fever duration/severity, travel history, "
        "geographic site risks, rashes, and signs of infection like stiff neck or bulging fontanelle. "
        "Your role is to identify bacterial, viral, or parasitic causes of death."
    )
    return {"agent1_output": _call_llm(state["full_dossier"], "specialist_id", focus)}

def agent2_node(state: VAState) -> dict:
    focus = (
        "Focus your analysis on acute physiological failure and systemic collapse. Look closely at "
        "respiratory distress (fast breathing, grunting, chest indrawing), neurological decline (unconsciousness, convulsions), "
        "and multi-organ system failure. Your role is to identify the acute pathway to death."
    )
    return {"agent2_output": _call_llm(state["full_dossier"], "specialist_cc", focus)}

def agent3_node(state: VAState) -> dict:
    focus = (
        "Focus your analysis on external, environmental, and baseline physical factors. Look closely at "
        "reported injuries (bites, falls, fires), accidental history, and nutritional status indicators "
        "(weight-for-age, edema, stunting/wasting). Your role is to identify non-infectious causes or "
        "underlying physical vulnerabilities."
    )
    return {"agent3_output": _call_llm(state["full_dossier"], "specialist_tn", focus)}
