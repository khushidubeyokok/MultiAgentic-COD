"""
agents/stage1.py
----------------
Stage 1: Broad Triage Classifier.
Groups dossiers into one of three broad categories to narrow the specialist's search space.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import strip_thoughts, parse_best_json
from agents.few_shot_examples import format_few_shot_block

# ── Global Few-Shot Library ──────────────────────────────────────────────────
# Initialized by run_pipeline.py on startup
FEW_SHOT_LIBRARY = {}

_LLM = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0,
    num_ctx=8192,
    num_predict=4096,  # Increased from 512 to prevent truncation
)

_STAGE1_SYSTEM = (
    "You are a triage classifier. You MUST follow these RULES in order:\n"
    "1. RULE 1 (EXTERNAL PRIORITY): If the dossier mentions ANY bite, sting, fall, fire, drowning, collision, poisoning, or violence — "
    "even if the patient has medical symptoms (fever, seizures) — you MUST classify as 'External/Trauma'.\n"
    "2. RULE 2 (CHRONICITY): If no Rule 1 condition exists AND the illness lasted weeks/months with wasting or HIV markers → 'Chronic/Systemic/Other'.\n"
    "3. RULE 3: Everything else → 'Infectious/Disease'.\n\n"
    "Output ONLY a JSON object: {\"broad_group\": \"...\", \"triage_reasoning\": \"...\"}"
)

_STAGE1_PROMPT = """### TARGET GROUPS ###
GROUP A: External/Trauma
GROUP B: Infectious/Disease
GROUP C: Chronic/Systemic/Other

### TASK ###
Classify the following dossier. 
IMPORTANT: Check the [Injury/Accident] section. If it reports a bite or sting, use GROUP A.

### PATIENT DOSSIER ###
{dossier}
"""

def stage1_node(state: VAState) -> dict:
    dossier = state["full_dossier"]
    
    # We remove few-shot for triage to prevent the model from over-relying on similar symptoms 
    # and ignoring the binary triage rules.
    prompt = _STAGE1_PROMPT.format(dossier=dossier[:12000])

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
