"""
agents/adjudicator.py
---------------------
Final adjudicator node. Uses the structured critic verdict to short-circuit
or guide the LLM call.

Consensus path  → return immediately (confidence 95), no LLM call needed.
Split path      → pass the critic's strongest_agent + recommended_diagnosis
                  as a strong prior hint to the LLM.
"""

import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, strip_thoughts, PHMRC_CATEGORIES

_LLM = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    num_ctx=8192,
)

PHMRC_LIST = ", ".join(PHMRC_CATEGORIES)

_ADJUDICATOR_SYSTEM = (
    "You are the Final Adjudicator — the last decision-maker in a structured multi-agent "
    "diagnostic panel. You have access to all specialist reasoning and a structured clinical "
    "critique. Your job is to render a single, authoritative final diagnosis."
)


def _parse_critique(critique_str: str) -> dict | None:
    """
    Try to parse the critic output as structured JSON.
    Returns dict if successful and has a 'verdict' key, else None.
    """
    if not critique_str:
        return None
    try:
        obj = json.loads(critique_str)
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    except Exception:
        pass
    # Try via parse_best_json as a fallback
    obj = parse_best_json(critique_str)
    if obj and "verdict" in obj:
        return obj
    return None


def _build_split_prompt(state: VAState, critic_verdict: dict | None) -> str:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]

    def _get(d: dict, k: str) -> str:
        return str(d.get(k, "Unknown"))

    critic_hint = ""
    if critic_verdict and critic_verdict.get("verdict") == "split":
        strongest = critic_verdict.get("strongest_agent", "")
        recommended = critic_verdict.get("recommended_diagnosis", "")
        reason = critic_verdict.get("reason", "")
        critic_hint = (
            f"\n### CLINICAL CRITIC RECOMMENDATION ###\n"
            f"The clinical critic recommends '{recommended}' based on: {reason}\n"
            f"(Strongest reasoning agent: {strongest})\n"
            f"Evaluate whether you agree. If so, confirm it. "
            f"Only override if you find a clear error in the critic's reasoning.\n"
        )

    critique_display = state.get("critique", "No critique available.")

    return f"""### PATIENT DOSSIER ###
{state["full_dossier"][:12000]}

### SPECIALIST INPUTS ###
Agent 1 (Evidence Collector): {_get(a1, "diagnosis")} | Reasoning: {_get(a1, "primary_reasoning")}
Agent 2 (Eliminator): {_get(a2, "diagnosis")} | Reasoning: {_get(a2, "primary_reasoning")}
Agent 3 (Timeline Analyst): {_get(a3, "diagnosis")} | Reasoning: {_get(a3, "primary_reasoning")}
{critic_hint}
### FULL CRITIC REPORT ###
{critique_display}

### TASK ###
Render a final adjudication. Output a single JSON object:
{{"final_diagnosis": "text", "mapped_category": "EXACT_CATEGORY", "confidence_score": 0-100, "final_reasoning": "text", "winning_agent": "name"}}

Valid categories: {PHMRC_LIST}

⚠️ CONSTRAINT: Your mapped_category MUST be one of the diagnoses already proposed by the three agents or the critic's recommended_diagnosis. Do NOT output a different category that no agent argued for. If agents disagree, pick the one whose reasoning you find most internally consistent with the dossier — do not hedge to a vague catch-all.

After the JSON, repeat: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]"""


def adjudicator_node(state: VAState) -> dict:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]

    # ── PARSE CRITIC VERDICT ───────────────────────────────────────────────────
    critique_str = state.get("critique", "")
    critic_verdict = _parse_critique(critique_str)

    # ── CONSENSUS FAST-PATH (driven by structured critic verdict) ─────────────
    if critic_verdict and critic_verdict.get("verdict") == "consensus":
        category = critic_verdict.get("consensus_diagnosis", "")
        # Validate against known categories
        if category in PHMRC_CATEGORIES:
            return {
                "final_diagnosis": category,
                "mapped_category": category,
                "confidence_score": 95,
                "final_reasoning": "Clinical critic confirmed unanimous consensus among all three specialist agents.",
                "winning_agent": "Consensus (all agents)",
            }

    # ── LEGACY CONSENSUS FORCE (agent outputs agree, critic may not have parsed) ─
    d1, d2, d3 = a1.get("diagnosis"), a2.get("diagnosis"), a3.get("diagnosis")
    if d1 == d2 == d3 and d1 in PHMRC_CATEGORIES:
        return {
            "final_diagnosis": str(d1),
            "mapped_category": str(d1),
            "confidence_score": 95,
            "final_reasoning": "Unanimous consensus among all three specialist agents.",
            "winning_agent": "Consensus (all agents)",
        }

    # ── LLM ADJUDICATION ──────────────────────────────────────────────────────
    prompt = _build_split_prompt(state, critic_verdict)
    response = _LLM.invoke([
        SystemMessage(content=_ADJUDICATOR_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)

    result = parse_best_json(raw_text)

    cat = result.get("mapped_category", result.get("diagnosis", "Other Defined Causes of Child Deaths"))

    # Remap if needed
    if cat not in PHMRC_CATEGORIES:
        for c in PHMRC_CATEGORIES:
            if c.lower() in str(cat).lower():
                cat = c
                break
        else:
            cat = "Other Defined Causes of Child Deaths"

    return {
        "final_diagnosis":  str(result.get("final_diagnosis", result.get("diagnosis", cat))),
        "mapped_category":  str(cat),
        "confidence_score": int(result.get("confidence_score", 50)),
        "final_reasoning":  str(result.get("final_reasoning", result.get("primary_reasoning", "No reasoning provided."))),
        "winning_agent":    str(result.get("winning_agent", "Adjudicator Override")),
    }
