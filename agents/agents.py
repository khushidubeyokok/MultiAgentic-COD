"""
agents/agents.py
----------------
Defines the three specialist agent nodes with genuinely different reasoning protocols.

Agent 1 — The Evidence Collector 
  Protocol: bottom-up from raw symptoms → syndromes → category

Agent 2 — The Eliminator 
  Protocol: top-down elimination, start from all 21, rule out, pick best fit

Agent 3 — The Timeline Analyst 
  Protocol: temporal reasoning through the clinical narrative arc
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, PHMRC_CATEGORIES, PHMRC_CATEGORY_GUIDE
from agents.few_shot_examples import format_few_shot_block

# ── Global Few-Shot Library ──────────────────────────────────────────────────
# Initialized by run_pipeline.py on startup
FEW_SHOT_LIBRARY = {}

# ── Model config ──────────────────────────────────────────────────────────────
# Change _MODEL here to switch for all three agents at once.
# qwen2.5:7b  → ~4.5 GB RAM  (fits on most machines)
# qwen2.5:14b → ~9 GB RAM    (needs a machine with ≥10 GB VRAM free)
_MODEL = "deepseek-r1:8b"

_LLM = ChatOllama(
    model=_MODEL,
    temperature=0,
    num_ctx=8192,
    num_predict=4096,
)

PHMRC_LIST = ", ".join(PHMRC_CATEGORIES)

_CATEGORY_HEADER = f"\n\n### CATEGORY REFERENCE ###\n{PHMRC_CATEGORY_GUIDE}\n"

# ── Confusion guard — appended to every agent before JSON output ───────────────
_CONFUSION_GUARD = """
### BEFORE YOU OUTPUT — CHECK THESE COMMON MISTAKES ###
1. SEPSIS TRAP: Are you choosing Sepsis only because you see fever + deterioration?
   If yes, re-examine: Is the patient from Africa? → Consider Malaria first.
   Are there respiratory symptoms? → Consider Pneumonia.
   Is there neck stiffness? → Consider Meningitis.
   Sepsis is the answer only when nothing else fits.

2. MENINGITIS TRAP: Is stiff neck EXPLICITLY stated in the dossier?
   If NO stiff neck documented → Do NOT diagnose Meningitis.
   Choose Encephalitis (if seizures + altered consciousness) or Sepsis instead.

3. HEMORRHAGIC FEVER TRAP: Is there spontaneous bleeding from MULTIPLE sites?
   If only ONE bleeding site, or fever without confirmed multi-site bleeding → Not Hemorrhagic fever.

4. MALARIA REMINDER: Is the patient from sub-Saharan Africa with undifferentiated fever?
   If yes → Malaria is your first candidate, not Sepsis.

5. OTHER DIGESTIVE TRAP: Is the primary complaint abdominal pain WITHOUT diarrhea?
   Jaundice, vomiting alone, or colicky pain without loose stools = Other Digestive Diseases, not Sepsis.
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1 — THE EVIDENCE COLLECTOR 
# Protocol: bottom-up from symptoms → syndromes → category
# ──────────────────────────────────────────────────────────────────────────────

_AGENT1_SYSTEM = (
    "You are The Evidence Collector for a pediatric cause of death determination pipeline."
    "You reason strictly bottom-up: from raw clinical evidence to diagnosis. "
    "You never guess first and work backward; you always build up from the facts."
)

_AGENT1_PROTOCOL = """⚠️ IMPORTANT BIAS WARNING: Pneumonia is one of 21 possible categories and should only
be chosen if respiratory symptoms are CLEARLY PRIMARY and ACUTE (sudden onset, hours to days).
Do NOT default to Pneumonia simply because a cough or breathing difficulty is mentioned
alongside another primary illness. If the child has been sick for weeks, or has chronic
markers (wasting, thrush, recurrent illness), consider other categories first.

### REASONING PROTOCOL ###
Follow these steps in order. Do not skip any.

STEP 1 — SYMPTOM INVENTORY:
List EVERY symptom, clinical sign, lab finding, and physical observation mentioned in the dossier. Be exhaustive.

STEP 2 — SYNDROME GROUPING:
Group those symptoms into a recognizable clinical syndrome (e.g., "febrile illness with rash", "acute respiratory syndrome", "gastroenteritis with dehydration", "meningeal syndrome").

STEP 2b — PRIMACY CHECK:
Which symptom or symptom cluster most likely represents the PRIMARY presenting complaint — i.e., what the child was most likely brought to hospital for?
⚠️ CRITICAL: In infants and young children, convulsions, stiff neck, and labored breathing can be SECONDARY COMPLICATIONS of severe dehydration from diarrhea, or of metabolic disturbance — NOT necessarily independent CNS or respiratory disease. If prominent diarrhea is present alongside neurological or respiratory signs, do NOT automatically anchor on the neurological/respiratory syndrome. Ask: could the diarrhea be the primary driver and the other signs be complications?

⚠️ AIDS CHECK: Before finalizing any diagnosis, explicitly scan the dossier for these AIDS markers:
  - Oral thrush or mouth sores
  - Mother known HIV-positive
  - Recurrent infections (especially repeated pneumonia episodes)
  - Chronic diarrhea lasting >1 month
  - Failure to thrive or wasting over months
  - Persistent swollen lymph nodes
If 2 or more of these markers are present in the dossier, you MUST seriously consider AIDS
over any acute infectious diagnosis (Pneumonia, Sepsis, Meningitis). AIDS is a chronic
wasting illness, not a sudden acute event.

STEP 3 — CATEGORY MATCH:
Using the CATEGORY REFERENCE below, match the PRIMARY syndrome (from Step 2b) to the most fitting PHMRC category. Be specific about which clinical features drove your match.

STEP 4 — ALTERNATIVE & REJECTION:
Name exactly one alternative PHMRC category you considered, and give one clear reason why you are rejecting it in favour of your primary answer.

STEP 5 — JSON OUTPUT:
Before writing your JSON, revisit the confusion guard below.
"""
_AGENT1_PROTOCOL += _CONFUSION_GUARD
_AGENT1_PROTOCOL += """
Output a single JSON object in this exact schema — nothing before or after it:
{"diagnosis": "exact category name", "confidence": "High/Medium/Low", "primary_reasoning": "concise chain of evidence → syndrome → category", "alternative_rejected": "category name", "rejection_reason": "one sentence"}

After the JSON, repeat the diagnosis inside tags: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2 — THE SYMPTOM SCORER 
# Protocol: binary checklist scoring across all 21 categories
# ──────────────────────────────────────────────────────────────────────────────

_AGENT2_SYSTEM = (
    "You are a structured clinical checklist evaluator. You do not guess diagnoses. "
    "You answer binary questions about what is and is not present in a dossier, "
    "then score categories mechanically."
)

_AGENT2_PROTOCOL = f"""### REASONING PROTOCOL ###
1. Read the dossier carefully.
2. Go through the following fixed checklist. Answer Y or N for EVERY item based ONLY on explicit documentation in the dossier.
3. For each category, tally its score (1 point per 'Y').
4. The category with the highest score is your primary diagnosis.

### THE CHECKLIST ###
- Respiratory (→ Pneumonia): acute cough present? fast/difficult breathing primary complaint? chest indrawing? fever present? sudden onset (hours to days)?
- Fever-Africa (→ Malaria): spiking or cyclical fever? patient in sub-Saharan Africa or endemic region? anaemia or splenomegaly? no clear bacterial source?
- Meningeal (→ Meningitis): stiff neck EXPLICITLY documented? bulging fontanelle? photophobia? Kernig/Brudzinski signs?
- CNS-no-neck (→ Encephalitis): seizures present? altered consciousness? fever? stiff neck ABSENT?
- Multi-organ fever (→ Sepsis): rapid multi-organ deterioration? fever without single focal site? Africa ruled out? Pneumonia/Meningitis ruled out?
- GI-primary (→ Diarrhea/Dysentery): watery or bloody stools as PRIMARY complaint? severe dehydration? sunken eyes?
- Rash (→ Measles): maculopapular rash explicitly documented? face-to-body spread? fever + cough + conjunctivitis?
- AIDS-markers (→ AIDS): oral thrush? mother HIV+? recurrent infections? chronic diarrhea >1 month? wasting over months?
- Multi-bleed (→ Hemorrhagic fever): spontaneous bleeding from 2+ sites? fever present simultaneously?
- Submersion (→ Drowning): found in/near water? submersion reported? water in airways?
- Vehicle (→ Road Traffic): vehicle collision mentioned? road accident? blunt trauma from impact?
- Height (→ Falls): fall from height reported? child found injured after fall?
- Burn (→ Fires): burn injuries? fire/flame exposure? smoke inhalation?
- Force (→ Violent Death): injuries inconsistent with history? assault documented? abuse signs?
- Toxic (→ Poisonings): toxic substance ingestion? vomiting after exposure? no other explanation?
- Envenomation (→ Bite of Venomous Animal): snake or scorpion bite reported? local swelling/necrosis? systemic toxicity?
- Cardiac (→ Other Cardiovascular): murmur? cyanosis? oedema as primary? arrhythmia? no infection primary?
- Mass (→ Other Cancers): chronic illness weeks to months? palpable mass? unexplained weight loss? no fever pattern?
- Abdominal-no-diarrhea (→ Other Digestive): abdominal pain/jaundice WITHOUT diarrhea as primary? vomiting only? colicky pain in infant?
- Pathogen-no-fit (→ Other Infectious Diseases): confirmed infection (typhoid/TB/pertussis) not fitting above categories?
- Neonatal/Congenital (→ Other Defined Causes): prematurity? birth asphyxia? congenital anomaly? neonatal period?

### JSON OUTPUT SCHEMA ###
Output a single JSON object (no markdown, no preamble):
{{
  "agent_name": "agent2_symptom_scorer",
  "diagnosis": "<top scored category name>",
  "confidence": "High/Medium/Low",
  "primary_reasoning": "<one sentence summary of the scoring>",
  "scores": {{
    "Pneumonia": <int>, "Malaria": <int>, "Meningitis": <int>, "Encephalitis": <int>, "Sepsis": <int>,
    "Diarrhea/Dysentery": <int>, "Measles": <int>, "AIDS": <int>, "Hemorrhagic fever": <int>,
    "Drowning": <int>, "Road Traffic": <int>, "Falls": <int>, "Fires": <int>, "Violent Death": <int>,
    "Poisonings": <int>, "Bite of Venomous Animal": <int>, "Other Cardiovascular Diseases": <int>,
    "Other Cancers": <int>, "Other Digestive Diseases": <int>, "Other Infectious Diseases": <int>,
    "Other Defined Causes of Child Deaths": <int>
  }},
  "top3": ["Cat1", "Cat2", "Cat3"],
  "checklist_notes": "<key positive findings that drove the top score>"
}}
"""
_AGENT2_PROTOCOL += _CONFUSION_GUARD

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3 — THE TIMELINE ANALYST 
# Protocol: temporal reasoning through the clinical narrative
# ──────────────────────────────────────────────────────────────────────────────

_AGENT3_SYSTEM = (
    "You are The Timeline Analyst for a pediatric cause of death determination pipeline. "
    "You reason temporally. You read a dossier like a detective reading a case file: "
    "you reconstruct the chronological story of the illness from baseline to death, "
    "and you let the trajectory — not just the snapshot — drive your diagnosis."
)

_AGENT3_PROTOCOL = """⚠️ IMPORTANT BIAS WARNING: Pneumonia is one of 21 possible categories and should only
be chosen if respiratory symptoms are CLEARLY PRIMARY and ACUTE (sudden onset, hours to days).
Do NOT default to Pneumonia simply because a cough or breathing difficulty is mentioned
alongside another primary illness. If the child has been sick for weeks, or has chronic
markers (wasting, thrush, recurrent illness), consider other categories first.

### REASONING PROTOCOL ###
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
Before writing your JSON, revisit the confusion guard below.
"""
_AGENT3_PROTOCOL += _CONFUSION_GUARD
_AGENT3_PROTOCOL += """
Output a single JSON object in this exact schema — nothing before or after it:
{"diagnosis": "exact category name", "confidence": "High/Medium/Low", "primary_reasoning": "concise temporal narrative that justifies the category", "timeline_duration": "e.g. acute <72h / subacute 3-14d / chronic >2wk", "nutritional_modifier": "Yes/No — brief note if relevant"}

After the JSON, repeat the diagnosis inside tags: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]
"""

# ── Category Groups ───────────────────────────────────────────────────────────
_GROUPS = {
    "External/Trauma": [
        "Drowning", "Falls", "Fires", "Road Traffic", "Violent Death", 
        "Poisonings", "Bite of Venomous Animal"
    ],
    "Infectious/Disease": [
        "Pneumonia", "Malaria", "Meningitis", "Encephalitis", "Sepsis", 
        "Measles", "AIDS", "Other Infectious Diseases", "Hemorrhagic fever", 
        "Diarrhea/Dysentery"
    ],
    "Chronic/Systemic/Other": [
        "Other Cancers", "Other Defined Causes of Child Deaths",
        "Other Digestive Diseases", "Other Cardiovascular Diseases"
    ]
}

# ──────────────────────────────────────────────────────────────────────────────
# Internal LLM caller
# ──────────────────────────────────────────────────────────────────────────────

def _call_llm(dossier: str, agent_key: str, system_msg: str, protocol_prompt: str, broad_group: str = "") -> dict:
    triage_context = ""
    target_cats = PHMRC_CATEGORIES
    if broad_group in _GROUPS:
        target_cats = _GROUPS[broad_group]
        cats_str = ", ".join(target_cats)
        triage_context = (
            f"⚠️ TRIAGE CONTEXT: This case has been pre-classified as: {broad_group}\n"
            f"Your candidate categories are therefore LIMITED TO: {cats_str}\n\n"
            f"DO NOT diagnose a category outside this group unless the evidence is overwhelming and unambiguous. "
            f"If you genuinely believe the triage is wrong, still output from the listed categories "
            f"and note your concern in primary_reasoning.\n\n"
        )

    few_shot_block = format_few_shot_block(FEW_SHOT_LIBRARY, categories=target_cats)

    full_prompt = (
        f"{few_shot_block}\n"
        f"{triage_context}"
        f"{protocol_prompt}"
        f"{_CATEGORY_HEADER}"
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
    result = _call_llm(
        state["full_dossier"],
        "agent1_evidence_collector",
        _AGENT1_SYSTEM,
        _AGENT1_PROTOCOL,
        state.get("broad_group", "")
    )
    return {"agent1_output": result}


def agent2_node(state: VAState) -> dict:
    result = _call_llm(
        state["full_dossier"],
        "agent2_symptom_scorer",
        _AGENT2_SYSTEM,
        _AGENT2_PROTOCOL,
        state.get("broad_group", "")
    )
    return {"agent2_output": result}


def agent3_node(state: VAState) -> dict:
    result = _call_llm(
        state["full_dossier"],
        "agent3_timeline_analyst",
        _AGENT3_SYSTEM,
        _AGENT3_PROTOCOL,
        state.get("broad_group", "")
    )
    return {"agent3_output": result}
