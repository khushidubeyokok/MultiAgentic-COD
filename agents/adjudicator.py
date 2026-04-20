"""
agents/adjudicator.py
---------------------
Final adjudicator node. Weighs specialist agent inputs and renders a final verdict.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, strip_thoughts, fuzzy_match_category, GEMMA4_THINKING_PREFIX
from agents.model_config import make_llm, ACTIVE_PROFILE
from agents.disease_ref import get_full_disease_ref

_LLM = make_llm()

_ADJUDICATOR_SYSTEM = GEMMA4_THINKING_PREFIX + "You are the final diagnostic arbitrator. Three specialist agents have each analysed a patient dossier and given you their diagnosis and reasoning. Analyze the top 3 categories returned by Agent 2 (Symptom Scorer) specifically. Your job is to weigh their input alongside the dossier and render one final verdict. Output only JSON."

def _build_adjudication_prompt(state: VAState) -> str:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]
    dossier = state["full_dossier"]
    full_disease_ref = get_full_disease_ref()

    def _get(d: dict, k: str) -> str:
        return str(d.get(k, "Unknown"))

    return f"""Section 1 — The three agent inputs:

Agent 1 (Evidence Collector): {_get(a1, "diagnosis")} | Confidence: {_get(a1, "confidence")}
Reasoning: {_get(a1, "primary_reasoning")}
Alternative rejected: {_get(a1, "alternative_rejected")} because {_get(a1, "rejection_reason")}

Agent 2 (Symptom Scorer): {_get(a2, "diagnosis")} | Confidence: {_get(a2, "confidence")}
Reasoning: {_get(a2, "primary_reasoning")}
Top 3 considered: {_get(a2, "top3")}

Agent 3 (Timeline Analyst): {_get(a3, "diagnosis")} | Confidence: {_get(a3, "confidence")}
Reasoning: {_get(a3, "primary_reasoning")}
Timeline: {_get(a3, "timeline_duration")}

Section 2 — The full dossier:
{dossier}

Section 3 — The full 21-category disease reference:
{full_disease_ref}

Section 4 — Instructions:
The three agents were given only the categories in the triage group. If you believe the triage was wrong and the correct category is from a different group, you may choose from the full list above. Otherwise prefer the agents' proposals. Give more weight to agents whose cited reasoning directly quotes or closely references specific findings from the dossier.

Section 5 — Output format:
```
{{"final_diagnosis": "<category>", "mapped_category": "<exact PHMRC category name>", "confidence_score": <0-100>, "final_reasoning": "<two sentences: which agents you agreed with and what dossier evidence drove the decision>", "winning_agent": "<agent1_evidence_collector / agent2_symptom_scorer / agent3_timeline_analyst / split>"}}
```

Section 6 — [FINAL_DIAGNOSIS] tag line.
"""

def consensus_node(state: VAState) -> dict:
    """
    Fast-path node for unanimous agent consensus.
    Bypasses Critic and Adjudicator LLMs.
    """
    a1 = state["agent1_output"]["diagnosis"]
    m1 = fuzzy_match_category(str(a1))

    # Use confidence score from active model profile
    conf_score = ACTIVE_PROFILE.get("consensus_confidence", 72)
    print(f"  [CONSENSUS] All three agents: {m1}. Confidence set to {conf_score} (profile: {ACTIVE_PROFILE['model']}).")
    
    return {
        "final_diagnosis":  str(a1),
        "mapped_category":  str(m1),
        "confidence_score": conf_score,
        "final_reasoning":  "Unanimous agent consensus",
        "winning_agent":    "all",
    }

def adjudicator_node(state: VAState) -> dict:
    prompt = _build_adjudication_prompt(state)
    response = _LLM.invoke([
        SystemMessage(content=_ADJUDICATOR_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)
    cleaned = strip_thoughts(raw_text)
    result = parse_best_json(cleaned)

    # Resolve mapped_category with fuzzy matching
    raw_cat = result.get("mapped_category", result.get("diagnosis", ""))
    cat = fuzzy_match_category(str(raw_cat)) if raw_cat else None

    if not cat:
        raw_fd = result.get("final_diagnosis", "")
        cat = fuzzy_match_category(str(raw_fd)) if raw_fd else None

    if not cat:
        cat = "Other Defined Causes of Child Deaths"

    return {
        "final_diagnosis":  str(result.get("final_diagnosis", result.get("diagnosis", cat))),
        "mapped_category":  str(cat),
        "confidence_score": int(result.get("confidence_score", 50)),
        "final_reasoning":  str(result.get("final_reasoning", result.get("primary_reasoning", "No reasoning provided."))),
        "winning_agent":    str(result.get("winning_agent", "Adjudicator Override")),
    }
