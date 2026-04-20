"""
agents/stage1.py
----------------
Stage 1: Broad Triage Classifier.
Groups dossiers into one of three broad categories to narrow the specialist's search space.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import strip_thoughts, parse_best_json, GEMMA4_THINKING_PREFIX
from agents.model_config import make_llm
from agents.disease_ref import get_full_disease_ref

_LLM = make_llm()

# New system prompt (under 40 words)
_STAGE1_SYSTEM = GEMMA4_THINKING_PREFIX + "You are a medical triage classifier. Read the patient dossier and assign it to exactly one of three groups based on the dominant cause of illness. Output only JSON."

_STAGE1_USER_PROMPT_TEMPLATE = """### GROUP DEFINITIONS ###

- Infectious/Disease: Includes Pneumonia, Malaria, Meningitis, Encephalitis, Sepsis, Diarrhea/Dysentery, Measles, AIDS, Hemorrhagic fever, and Other Infectious Diseases. This group unifies illnesses caused by pathogens that typically present with fever and acute or chronic physiological deterioration.

- External/Trauma: Includes Drowning, Road Traffic, Falls, Fires, Violent Death, Poisonings, and Bite of Venomous Animal. This group unifies deaths caused by physical forces, accidents, or external agents acting on the body from the environment.

- Chronic/Systemic/Other: Includes Other Cardiovascular Diseases, Other Cancers, Other Digestive Diseases, and Other Defined Causes of Child Deaths. This group unifies non-infectious, long-term conditions or specific neonatal/congenital etiologies that do not fit the acute infectious or traumatic patterns.

### FULL 21-CATEGORY DISEASE REFERENCE ###
{disease_ref}

### CLASSIFICATION RULE ###
Classify by what CAUSED the illness, not what symptoms appeared. For example, if a child fell and later had seizures, the cause is External/Trauma.

### PATIENT DOSSIER ###
{dossier}

### OUTPUT INSTRUCTION ###
Output exactly this JSON format:
{{"broad_group": "External/Trauma" | "Infectious/Disease" | "Chronic/Systemic/Other", "triage_reasoning": "one sentence"}}
"""

def stage1_node(state: VAState) -> dict:
    dossier = state["full_dossier"]
    disease_ref = get_full_disease_ref()
    
    # Pass full dossier and full disease reference
    prompt = _STAGE1_USER_PROMPT_TEMPLATE.format(
        dossier=dossier,
        disease_ref=disease_ref
    )

    response = _LLM.invoke([
        SystemMessage(content=_STAGE1_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)
    cleaned = strip_thoughts(raw_text)
    
    parsed = parse_best_json(cleaned)
    group = parsed.get("broad_group", "Infectious/Disease")
    
    # Validation
    valid_groups = ["External/Trauma", "Infectious/Disease", "Chronic/Systemic/Other"]
    if group not in valid_groups:
        group = "Infectious/Disease"
        
    return {"broad_group": group}
