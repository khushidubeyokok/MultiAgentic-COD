"""
agents/agents.py
----------------
Defines the three specialist agent nodes with genuinely different reasoning protocols.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, GEMMA4_THINKING_PREFIX
from agents.model_config import make_llm
from agents.disease_ref import get_disease_ref

# Instantiate LLM once at module level
_LLM = make_llm()

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1 — THE EVIDENCE COLLECTOR 
# ──────────────────────────────────────────────────────────────────────────────

_AGENT1_SYSTEM = GEMMA4_THINKING_PREFIX + "You are a clinical evidence collector. You read patient dossiers and identify cause of death by working bottom-up from documented symptoms to the best matching disease category. Output only JSON."

_AGENT1_PROTOCOL = """Section 1 — Triage context placeholder:
{triage_context}

Section 2 — Reasoning approach, exactly 4 bullet points, no sub-bullets:
- List every symptom, sign, and finding explicitly documented in the dossier
- Identify which symptom or cluster is the PRIMARY complaint — what brought the child to care
- Match the primary complaint to the most fitting category from the list above
- Name one alternative you considered and one reason you rejected it

Section 3 — Output format:
```
{"agent_name": "agent1_evidence_collector", "diagnosis": "<exact category name>", "confidence": "High/Medium/Low", "primary_reasoning": "<two sentences: primary finding and why it maps to this category>", "alternative_rejected": "<category>", "rejection_reason": "<one sentence>"}
```

Section 4 — One line: After the JSON write: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2 — THE SYMPTOM SCORER 
# ──────────────────────────────────────────────────────────────────────────────

_AGENT2_SYSTEM = GEMMA4_THINKING_PREFIX + "You are a clinical checklist evaluator. You answer binary yes/no questions about a patient dossier, then score disease categories mechanically based on your answers. Output only JSON."

_AGENT2_PROTOCOL = """Section 1 — Triage context placeholder:
{triage_context}

Section 2 — Instructions, 3 sentences:
Read the dossier. Answer every checklist question with yes or no based only on what is explicitly written. Tally scores and pick the highest.

Section 3 — The checklist:
[Pneumonia]: acute cough present? fast/difficult breathing primary complaint? chest indrawing? fever present? sudden onset?
[Malaria]: spiking or cyclical fever? patient in sub-Saharan Africa or endemic region? anaemia or splenomegaly? no clear bacterial source?
[Meningitis]: stiff neck EXPLICITLY documented? bulging fontanelle? photophobia? Kernig/Brudzinski signs?
[Encephalitis]: seizures present? altered consciousness? fever? stiff neck ABSENT?
[Sepsis]: rapid multi-organ deterioration? fever without single focal site? Africa ruled out? Pneumonia/Meningitis ruled out?
[Diarrhea/Dysentery]: watery or bloody stools as PRIMARY complaint? severe dehydration? sunken eyes?
[Measles]: maculopapular rash explicitly documented? face-to-body spread? fever + cough + conjunctivitis?
[AIDS]: oral thrush? mother HIV+? recurrent infections? chronic diarrhea >1 month? wasting over months?
[Hemorrhagic fever]: spontaneous bleeding from 2+ sites? fever present simultaneously?
[Drowning]: found in/near water? submersion reported? water in airways?
[Road Traffic]: vehicle collision mentioned? road accident? blunt trauma from impact?
[Falls]: fall from height reported? child found injured after fall?
[Fires]: burn injuries? fire/flame exposure? smoke inhalation?
[Violent Death]: injuries inconsistent with history? assault documented? abuse signs?
[Poisonings]: toxic substance ingestion? vomiting after exposure? no other explanation?
[Bite of Venomous Animal]: snake or scorpion bite reported? local swelling/necrosis? systemic toxicity?
[Other Cardiovascular Diseases]: murmur? cyanosis? oedema as primary? arrhythmia? no infection primary?
[Other Cancers]: chronic illness weeks to months? palpable mass? unexplained weight loss? no fever pattern?
[Other Digestive Diseases]: abdominal pain/jaundice WITHOUT diarrhea as primary? vomiting only? colicky pain in infant?
[Other Infectious Diseases]: confirmed infection (typhoid/TB/pertussis) not fitting above categories?
[Other Defined Causes of Child Deaths]: prematurity? birth asphyxia? congenital anomaly? neonatal period?

Section 4 — Output format:
```
{"agent_name": "agent2_symptom_scorer", "diagnosis": "<top scored category>", "confidence": "High/Medium/Low", "primary_reasoning": "<one sentence: top score was X with these key positives>", "top3": ["Cat1", "Cat2", "Cat3"]}
```

Section 5 — [FINAL_DIAGNOSIS] tag line.
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3 — THE TIMELINE ANALYST 
# ──────────────────────────────────────────────────────────────────────────────

_AGENT3_SYSTEM = GEMMA4_THINKING_PREFIX + "You are a clinical timeline analyst. You reconstruct the chronological story of a child's illness from baseline to death and identify cause of death from the trajectory. Output only JSON."

_AGENT3_PROTOCOL = """Section 1 — Triage context placeholder:
{triage_context}

Section 2 — Reasoning approach, exactly 4 bullet points:
- What was the child's baseline health before illness began
- What was the first sign of illness and how long before death did it appear
- How did the illness progress — was it rapid (hours/days) or slow (weeks/months)
- Which category best matches this complete trajectory from onset to death

Section 3 — Output format:
```
{"agent_name": "agent3_timeline_analyst", "diagnosis": "<exact category name>", "confidence": "High/Medium/Low", "primary_reasoning": "<two sentences: timeline summary and why it maps to this category>", "timeline_duration": "acute <72h / subacute 3-14d / chronic >2wk"}
```

Section 4 — [FINAL_DIAGNOSIS] tag line.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Internal LLM caller
# ──────────────────────────────────────────────────────────────────────────────

def _call_llm(dossier: str, agent_key: str, system_msg: str, protocol_prompt: str, broad_group: str) -> dict:
    # 1. Call get_disease_ref(broad_group)
    disease_list_text = get_disease_ref(broad_group)

    # 2. Build triage_context
    triage_context = (
        f"Broad Group: {broad_group}\n"
        f"Your diagnosis must come from this list unless you have strong evidence the triage was wrong:\n"
        f"{disease_list_text}"
    )

    # 3. Inject triage_context into the protocol string
    injected_protocol = protocol_prompt.replace("{triage_context}", triage_context)

    # 4. Build final prompt
    full_prompt = injected_protocol + "\n\n### PATIENT DOSSIER ###\n" + dossier

    response = _LLM.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=full_prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = parse_best_json(raw_text)
    
    if not parsed or ("diagnosis" not in parsed and "broad_group" not in parsed) or parsed.get("diagnosis") == "Unknown":
        print(f"[WARN] {agent_key}: Valid diagnosis not found in JSON. Returning error dict.")
        return {
            "agent_name": agent_key,
            "diagnosis": "Unknown",
            "confidence": "Low",
            "primary_reasoning": "Reasoning model failed to output a valid diagnosis key.",
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
        state.get("broad_group", "Infectious/Disease")
    )
    return {"agent1_output": result}


def agent2_node(state: VAState) -> dict:
    result = _call_llm(
        state["full_dossier"],
        "agent2_symptom_scorer",
        _AGENT2_SYSTEM,
        _AGENT2_PROTOCOL,
        state.get("broad_group", "Infectious/Disease")
    )
    return {"agent2_output": result}


def agent3_node(state: VAState) -> dict:
    result = _call_llm(
        state["full_dossier"],
        "agent3_timeline_analyst",
        _AGENT3_SYSTEM,
        _AGENT3_PROTOCOL,
        state.get("broad_group", "Infectious/Disease")
    )
    return {"agent3_output": result}
