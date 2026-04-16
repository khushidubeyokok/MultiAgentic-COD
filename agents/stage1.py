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
    num_predict=512,
)

_STAGE1_SYSTEM = (
    "You are a triage classifier. You read a patient dossier and assign it to "
    "exactly one of three broad cause-of-death groups. You do not diagnose; you group."
)

_STAGE1_PROMPT = """### BROAD GROUPS ###

GROUP A — External/Trauma:
  Drowning, Falls, Fires, Road Traffic, Violent Death, Poisonings, Bite of Venomous Animal

GROUP B — Infectious/Disease:
  Pneumonia, Malaria, Meningitis, Encephalitis, Sepsis, Measles, AIDS,
  Other Infectious Diseases, Hemorrhagic fever, Diarrhea/Dysentery

GROUP C — Chronic/Systemic/Other:
  Other Cancers, Other Defined Causes of Child Deaths,
  Other Digestive Diseases, Other Cardiovascular Diseases

### TRIAGE RULES ###
1. If ANY external event is reported (fall, bite, collision, fire, submersion, ingestion, assault) → External/Trauma, regardless of subsequent symptoms.
2. If the illness is chronic (weeks to months of decline, wasting, masses) with no external event → Chronic/Systemic/Other.
3. Everything else → Infectious/Disease.

### TASK ###
Classify the following dossier into exactly one of the three group names.
Output a single JSON object:
{{"broad_group": "External/Trauma" | "Infectious/Disease" | "Chronic/Systemic/Other", "triage_reasoning": "one sentence"}}

### PATIENT DOSSIER ###
{dossier}
"""

def stage1_node(state: VAState) -> dict:
    dossier = state["full_dossier"]
    
    # Pick one representative category from each group for triage grounding
    triage_examples = ["Drowning", "Pneumonia", "Other Cancers"]
    few_shot_block = format_few_shot_block(FEW_SHOT_LIBRARY, categories=triage_examples)

    prompt = (
        f"{few_shot_block}\n"
        f"{_STAGE1_PROMPT.format(dossier=dossier[:12000])}"
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
