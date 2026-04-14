"""
agents/agents.py
----------------
Three specialist agents with genuinely different reasoning protocols.

Agent 1 — The Evidence Collector: bottom-up from symptoms → syndromes → category
Agent 2 — The Eliminator: top-down elimination from all 21 categories
Agent 3 — The Timeline Analyst: temporal reasoning through narrative arc

Merged with: llm_config (multi-backend), preprocessor (condensed dossier),
dynamic few-shot examples.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, PHMRC_CATEGORIES, PHMRC_CATEGORY_GUIDE
from agents.llm_config import get_llm
from agents.few_shot_examples import get_examples_for_agent

_LLM = get_llm(temperature=0.0)

PHMRC_LIST = ", ".join(PHMRC_CATEGORIES)
_CATEGORY_HEADER = f"\n\n### CATEGORY REFERENCE ###\n{PHMRC_CATEGORY_GUIDE}\n"

# ── Clinical rules applied to ALL agents (fixes known failure patterns) ──────
_CLINICAL_RULES = """
### MANDATORY CLINICAL RULES ###
⚠️ You MUST follow these rules before outputting your diagnosis:

RULE 1 — "Other Infectious Diseases" IS A LAST RESORT:
Do NOT use "Other Infectious Diseases" unless you have explicitly ruled out ALL of these first: Pneumonia, Sepsis, Meningitis, Malaria, Measles, Diarrhea/Dysentery, Hemorrhagic fever, AIDS. If the case has fever + any specific organ involvement, it is almost certainly one of those — not OID.

RULE 2 — GEOGRAPHIC CONTEXT MATTERS:
Check the patient's LOCATION. If the child is from sub-Saharan Africa (Tanzania, Nigeria, etc.) or South/Southeast Asia (India, Philippines, etc.) AND has prolonged/cyclical fever → strongly consider Malaria before any other febrile illness.

RULE 3 — BLEEDING = HEMORRHAGIC FEVER:
If the dossier mentions bleeding from mouth, nose, gums, or skin AND fever → diagnose Hemorrhagic fever, NOT "Other Infectious Diseases" or Sepsis.

RULE 4 — DIARRHEA PRIMACY:
If diarrhea/loose stools are present alongside respiratory symptoms (fast breathing, difficult breathing), consider that the breathing problems may be CAUSED BY dehydration from diarrhea. In such cases, the primary diagnosis should be Diarrhea/Dysentery, not Pneumonia — UNLESS there is also cough with chest indrawing (which suggests true respiratory infection).

RULE 5 — PNEUMONIA REQUIRES RESPIRATORY EVIDENCE:
Do NOT diagnose Pneumonia unless there is clear cough AND (difficult breathing OR fast breathing OR chest indrawing). Fever alone is not enough. Breathing problems alone without cough may indicate dehydration, cardiac failure, or metabolic acidosis.
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1 — THE EVIDENCE COLLECTOR
# ──────────────────────────────────────────────────────────────────────────────

_AGENT1_SYSTEM = (
    "You are The Evidence Collector for a pediatric cause of death determination pipeline. "
    "You reason strictly bottom-up: from raw clinical evidence to diagnosis. "
    "You never guess first and work backward; you always build up from the facts."
)

_AGENT1_PROTOCOL = """### REASONING PROTOCOL ###
Follow these steps in order. Do not skip any.

STEP 1 — SYMPTOM INVENTORY:
List EVERY symptom, clinical sign, lab finding, and physical observation mentioned in the dossier. Be exhaustive.

STEP 2 — SYNDROME GROUPING:
Group those symptoms into a recognizable clinical syndrome (e.g., "febrile illness with rash", "acute respiratory syndrome", "gastroenteritis with dehydration", "meningeal syndrome").

STEP 2b — PRIMACY CHECK:
Which symptom or symptom cluster most likely represents the PRIMARY presenting complaint — i.e., what the child was most likely brought to hospital for?
⚠️ CRITICAL: In infants and young children, convulsions, stiff neck, and labored breathing can be SECONDARY COMPLICATIONS of severe dehydration from diarrhea, or of metabolic disturbance — NOT necessarily independent CNS or respiratory disease. If prominent diarrhea is present alongside neurological or respiratory signs, do NOT automatically anchor on the neurological/respiratory syndrome. Ask: could the diarrhea be the primary driver and the other signs be complications?

STEP 3 — CATEGORY MATCH:
Using the CATEGORY REFERENCE below, match the PRIMARY syndrome (from Step 2b) to the most fitting PHMRC category. Be specific about which clinical features drove your match.

STEP 4 — ALTERNATIVE & REJECTION:
Name exactly one alternative PHMRC category you considered, and give one clear reason why you are rejecting it in favour of your primary answer.

STEP 5 — JSON OUTPUT:
Output a single JSON object in this exact schema — nothing before or after it:
{"diagnosis": "exact category name", "confidence": "High/Medium/Low", "primary_reasoning": "concise chain of evidence → syndrome → category", "alternative_rejected": "category name", "rejection_reason": "one sentence"}

After the JSON, repeat the diagnosis inside tags: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2 — THE ELIMINATOR
# ──────────────────────────────────────────────────────────────────────────────

_AGENT2_SYSTEM = (
    "You are The Eliminator for a pediatric cause of death determination pipeline. "
    "You reason top-down by systematic exclusion. "
    "Your starting position is: any of the 21 PHMRC categories could be the answer — "
    "your job is to aggressively rule out what is NOT supported and converge on what remains."
)

_AGENT2_PROTOCOL = f"""### REASONING PROTOCOL ###
Follow these steps in order. Do not skip any.

STEP 1 — MASS ELIMINATION:
From the full list of 21 PHMRC categories, immediately eliminate every category that requires evidence that is COMPLETELY ABSENT from the dossier. List each eliminated category and the single key piece of missing evidence that disqualifies it. Aim to eliminate at least 15.

⚠️ STRICT RULE: You may ONLY cite evidence that is EXPLICITLY STATED in the dossier. Do NOT infer, assume, or paraphrase symptoms that are not written there. If the dossier says "No cough reported" — cough is absent. Do not list it as present or potential evidence.

The 21 categories are: {PHMRC_LIST}

STEP 2 — SHORTLIST ANALYSIS:
From the 3–5 remaining categories, identify the best fit based on the most severe, definitive, or unambiguous clinical finding in the dossier.

STEP 3 — PIVOT TEST:
State what single piece of evidence, if present, would have changed your answer to a different category.

STEP 4 — JSON OUTPUT:
Output a single JSON object in this exact schema — nothing before or after it:
{{"diagnosis": "exact category name", "confidence": "High/Medium/Low", "primary_reasoning": "concise description of what survived elimination and why", "eliminated_count": <number>, "pivot_evidence": "what would have changed the answer"}}

After the JSON, repeat the diagnosis inside tags: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3 — THE TIMELINE ANALYST
# ──────────────────────────────────────────────────────────────────────────────

_AGENT3_SYSTEM = (
    "You are The Timeline Analyst for a pediatric cause of death determination pipeline. "
    "You reason temporally. You read a dossier like a detective reading a case file: "
    "you reconstruct the chronological story of the illness from baseline to death, "
    "and you let the trajectory — not just the snapshot — drive your diagnosis."
)

_AGENT3_PROTOCOL = """### REASONING PROTOCOL ###
Follow these steps in order. Do not skip any.

STEP 1 — BASELINE STATE:
What was the child's health status in the weeks or months BEFORE death? (Nutritional status, chronic illness, developmental history, household environment.)

STEP 2 — FIRST DETERIORATION:
What was the very first sign that something was wrong? How many days before death did it appear?

STEP 3 — PROGRESSION:
How did the illness evolve from that first sign to death? Was it rapid (hours/days) or slow (weeks)? Were there distinct phases?

STEP 4 — TERMINAL EVENT:
What appears to have been the immediate cause of death? What organ system failed last?

STEP 5 — CONTEXTUAL MODIFIERS:
Does the nutritional status, environmental context, or geographic setting meaningfully change the picture? Would the child have died from the same cause without those modifiers?

STEP 6 — CATEGORY FIT:
Based on this complete clinical trajectory, which PHMRC category best fits the overall story?

STEP 7 — JSON OUTPUT:
Output a single JSON object in this exact schema — nothing before or after it:
{"diagnosis": "exact category name", "confidence": "High/Medium/Low", "primary_reasoning": "concise temporal narrative that justifies the category", "timeline_duration": "e.g. acute <72h / subacute 3-14d / chronic >2wk", "nutritional_modifier": "Yes/No — brief note if relevant"}

After the JSON, repeat the diagnosis inside tags: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]
"""

# ──────────────────────────────────────────────────────────────────────────────
# Internal LLM caller
# ──────────────────────────────────────────────────────────────────────────────

def _call_llm(dossier: str, agent_key: str, system_msg: str, protocol_prompt: str) -> dict:
    # Get dynamic few-shot examples for this agent type
    agent_type = agent_key.split("_", 1)[0] if "_" in agent_key else agent_key
    # Map agent keys to specialist categories for few-shot
    persona_map = {
        "agent1": "specialist_id",
        "agent2": "specialist_cc",
        "agent3": "specialist_tn",
    }
    persona = persona_map.get(agent_type, "specialist_id")
    examples = get_examples_for_agent(persona)

    examples_section = ""
    if examples:
        examples_section = f"\n### REFERENCE EXAMPLES (from solved cases) ###\n{examples}\n"

    full_prompt = (
        f"{protocol_prompt}"
        f"{_CATEGORY_HEADER}"
        f"{_CLINICAL_RULES}"
        f"{examples_section}"
        f"\n### PATIENT DOSSIER ###\n{dossier}"
    )

    response = _LLM.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=full_prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = parse_best_json(raw_text)
    if not parsed:
        print(f"[WARN] {agent_key}: JSON parse failed. Returning error dict.")
        return {
            "agent_name": agent_key,
            "diagnosis": "Unknown",
            "confidence": "Low",
            "primary_reasoning": "Reasoning model output parse failed.",
            "error": True,
            "raw_response": raw_text,
        }

    parsed["agent_name"] = agent_key
    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph node functions
# ──────────────────────────────────────────────────────────────────────────────

def agent1_node(state: VAState) -> dict:
    dossier = state.get("condensed_dossier") or state["full_dossier"]
    return {"agent1_output": _call_llm(
        dossier, "agent1_evidence_collector", _AGENT1_SYSTEM, _AGENT1_PROTOCOL,
    )}


def agent2_node(state: VAState) -> dict:
    dossier = state.get("condensed_dossier") or state["full_dossier"]
    return {"agent2_output": _call_llm(
        dossier, "agent2_eliminator", _AGENT2_SYSTEM, _AGENT2_PROTOCOL,
    )}


def agent3_node(state: VAState) -> dict:
    dossier = state.get("condensed_dossier") or state["full_dossier"]
    return {"agent3_output": _call_llm(
        dossier, "agent3_timeline_analyst", _AGENT3_SYSTEM, _AGENT3_PROTOCOL,
    )}
