"""
agents/agents.py
----------------
Defines the three specialist agent nodes with genuinely different reasoning protocols.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json
from agents.model_config import make_llm
from agents.disease_ref import get_disease_ref, get_category_guide

# Instantiate LLM once at module level
_LLM = make_llm()

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1 — THE EVIDENCE COLLECTOR 
# ──────────────────────────────────────────────────────────────────────────────

_AGENT1_SYSTEM = "You are a clinical evidence collector identifying cause of death bottom-up from symptoms. Include a concise clinical rationale inside the JSON fields. Your final answer must be only valid JSON, with no markdown."

_AGENT1_PROTOCOL = """Section 1 — Triage context placeholder:
{triage_context}

Section 2 — Reasoning approach, exactly 4 bullet points, no sub-bullets:
- List every symptom, sign, and finding explicitly documented in the dossier
- Identify which symptom or cluster is the PRIMARY complaint — what brought the child to care
- Match the primary complaint to the most fitting category from the list above
- Name one alternative you considered and one reason you rejected it

Section 3 — Output format:
```
{"agent_name": "agent1_evidence_collector", "diagnosis": "<exact category name>", "confidence": "High/Medium/Low", "primary_reasoning": "<2-3 sentences: key evidence, primary complaint, and why it maps to this category>", "alternative_rejected": "<category>", "rejection_reason": "<one sentence>"}
```

Section 4 — Do not write anything before or after the JSON object.
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2 — THE SYMPTOM SCORER 
# ──────────────────────────────────────────────────────────────────────────────

_AGENT2_SYSTEM = "You are a clinical checklist evaluator scoring disease categories mechanically. Include a concise scoring rationale inside the JSON fields. Your final answer must be only valid JSON, with no markdown."

_AGENT2_PROTOCOL = """Section 1 - Triage context placeholder:
{triage_context}

Section 2 - Instructions:
Score only the categories in the triage list against the dossier. Pick the highest-scoring category and include the next two closest alternatives in top3. Do not use endemic location alone as a Malaria criterion; require malaria-specific evidence or absence of a stronger primary syndrome.

Section 3 - Output format:
```
{"agent_name": "agent2_symptom_scorer", "diagnosis": "<top scored category>", "confidence": "High/Medium/Low", "primary_reasoning": "<2-3 sentences: top score, key positives, and why close alternatives scored lower>", "top3": ["Cat1", "Cat2", "Cat3"]}
```

Section 4 - Do not write anything before or after the JSON object.
"""

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3 — THE TIMELINE ANALYST 
# ──────────────────────────────────────────────────────────────────────────────

_AGENT3_SYSTEM = "You are a clinical timeline analyst identifying cause of death from the disease trajectory. Include a concise timeline rationale inside the JSON fields. Your final answer must be only valid JSON, with no markdown."

_AGENT3_PROTOCOL = """Section 1 — Triage context placeholder:
{triage_context}

Section 2 — Reasoning approach, exactly 4 bullet points:
- What was the child's baseline health before illness began
- What was the first sign of illness and how long before death did it appear
- How did the illness progress — was it rapid (hours/days) or slow (weeks/months)
- Which category best matches this complete trajectory from onset to death

Section 3 — Output format:
```
{"agent_name": "agent3_timeline_analyst", "diagnosis": "<exact category name>", "confidence": "High/Medium/Low", "primary_reasoning": "<2-3 sentences: timeline summary, progression speed, and why it maps to this category>", "timeline_duration": "acute <72h / subacute 3-14d / chronic >2wk"}
```

Section 4 — Do not write anything before or after the JSON object.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Internal LLM caller
# ──────────────────────────────────────────────────────────────────────────────

def _call_llm(dossier: str, agent_key: str, system_msg: str, protocol_prompt: str, broad_group: str) -> dict:
    # 1. Call get_disease_ref(broad_group) and get_category_guide(broad_group)
    disease_list_text = get_disease_ref(broad_group)
    guide_text = get_category_guide(broad_group)

    # 2. Build triage_context
    triage_context = (
        f"Broad Group: {broad_group}\n"
        f"Your diagnosis must come from this list unless you have strong evidence the triage was wrong:\n"
        f"{disease_list_text}\n\n"
        f"### PRIMARY-CAUSE GUARDRAILS ###\n"
        f"Terminal deterioration, oxygen use, unconsciousness, inability to eat/drink, or multi-organ decline near death are complications, not automatically the cause category.\n"
        f"Do not choose Sepsis unless no more specific PHMRC category explains the initial and dominant illness syndrome.\n"
        f"Do not choose Malaria from fever plus geography alone; require the dossier pattern to fit malaria better than diarrhea, measles, pneumonia, CNS infection, or other infectious disease.\n\n"
        f"### CRITICAL DIAGNOSTIC GUIDELINES ###\n"
        f"{guide_text}"
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

    if not str(raw_text).strip():
        retry_prompt = (
            "Return exactly one valid JSON object and nothing else.\n"
            f"Allowed categories:\n{disease_list_text}\n\n"
            "JSON schema:\n"
            '{"diagnosis": "<exact category>", "confidence": "High/Medium/Low", '
            '"primary_reasoning": "<2 sentences>", "top3": ["Cat1", "Cat2", "Cat3"]}\n\n'
            f"PATIENT DOSSIER:\n{dossier}"
        )
        response = _LLM.invoke([
            SystemMessage(content="You are a medical verbal-autopsy classifier. Output only valid JSON."),
            HumanMessage(content=retry_prompt),
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
            "parse_failure": True,
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
