"""
agents/adjudicator.py
---------------------
Final adjudicator node. Uses the structured critic verdict to guide the LLM call.

CONSENSUS POLICY (non-blind):
  - Unanimous agent agreement OR critic "consensus" verdict is passed to the LLM
    as a STRONG PRIOR HINT — the adjudicator still performs independent reasoning
    and must explicitly confirm or override.
  - The blind "all-3-agree → instant return" fast-path has been removed.
    Critic and adjudicator always run their full reasoning step.

SPLIT POLICY:
  - Critic's strongest_agent + recommended_diagnosis is passed as a hint.
  - Adjudicator must pick from proposals already made by agents/critic.
"""

import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import parse_best_json, strip_thoughts, PHMRC_CATEGORIES, fuzzy_match_category

_LLM = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    num_ctx=8192,
)

PHMRC_LIST = ", ".join(PHMRC_CATEGORIES)

_ADJUDICATOR_SYSTEM = (
    "You are the Final Adjudicator — the last decision-maker in a structured multi-agent "
    "diagnostic panel. You have access to all specialist reasoning and a structured clinical "
    "critique. Your job is to render a single, authoritative final diagnosis. "
    "You must always reason from the evidence yourself — do NOT simply rubber-stamp what "
    "the majority agreed on without confirming it is supported by the dossier."
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


def _build_adjudication_prompt(state: VAState, critic_verdict: dict | None) -> str:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]

    def _get(d: dict, k: str) -> str:
        return str(d.get(k, "Unknown"))

    # ── Build consensus/split hint block ──────────────────────────────────────
    critic_hint = ""
    if critic_verdict:
        verdict_type = critic_verdict.get("verdict", "")
        recommended  = critic_verdict.get("recommended_diagnosis", "") or critic_verdict.get("consensus_diagnosis", "")
        strongest    = critic_verdict.get("strongest_agent", "")
        reason       = critic_verdict.get("reason", "")

        if verdict_type == "consensus":
            critic_hint = (
                f"\n### CRITIC CONSENSUS SIGNAL ###\n"
                f"The Clinical Critic reports that ALL THREE agents independently reached '{recommended}'.\n"
                f"Reason: {reason}\n"
                f"⚠️  IMPORTANT: This is a strong prior, but you MUST independently verify:\n"
                f"  (a) Does the dossier evidence actually support '{recommended}'?\n"
                f"  (b) Are the agents' reasoning chains internally consistent with the dossier?\n"
                f"  If both checks pass → confirm '{recommended}'. "
                f"If you find a clear error → override with your own diagnosis and explain why.\n"
            )
        elif verdict_type == "split":
            critic_hint = (
                f"\n### CRITIC SPLIT RECOMMENDATION ###\n"
                f"The Clinical Critic recommends '{recommended}' based on: {reason}\n"
                f"(Strongest reasoning agent: {strongest})\n"
                f"Evaluate whether you agree. Only override if you find a clear flaw.\n"
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
Render a final adjudication. Review the dossier and specialist inputs independently.
Output a single JSON object:
{{"final_diagnosis": "text", "mapped_category": "EXACT_CATEGORY", "confidence_score": 0-100, "final_reasoning": "text", "winning_agent": "name"}}

Valid categories: {PHMRC_LIST}

⚠️  CONSTRAINT: Your mapped_category MUST be one of the diagnoses already proposed by the three agents or the critic's recommended_diagnosis. Do NOT output a different category that no agent argued for. If agents disagree, pick the one whose reasoning is most internally consistent with the dossier — do not hedge to a vague catch-all.

After the JSON, repeat: [FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]"""


def adjudicator_node(state: VAState) -> dict:
    # ── PARSE CRITIC VERDICT ────────────────────────────────────────────────
    critique_str = state.get("critique", "")
    critic_verdict = _parse_critique(critique_str)

    # ── ALWAYS USE LLM ADJUDICATION ─────────────────────────────────────────
    # Consensus signals are passed as strong hints but never bypass reasoning.
    # This ensures critic and adjudicator always do their full job.
    prompt = _build_adjudication_prompt(state, critic_verdict)
    response = _LLM.invoke([
        SystemMessage(content=_ADJUDICATOR_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)

    result = parse_best_json(raw_text)

    # ── RESOLVE mapped_category with fuzzy matching ──────────────────────────
    raw_cat = result.get("mapped_category", result.get("diagnosis", ""))
    cat = fuzzy_match_category(str(raw_cat)) if raw_cat else None

    if not cat:
        # Last resort: try final_diagnosis field
        raw_fd = result.get("final_diagnosis", "")
        cat = fuzzy_match_category(str(raw_fd)) if raw_fd else None

    if not cat:
        cat = "Other Defined Causes of Child Deaths"

    # ── Determine winning_agent label ────────────────────────────────────────
    winning = result.get("winning_agent", "Adjudicator Override")
    if critic_verdict and critic_verdict.get("verdict") == "consensus":
        # Preserve the fact that this was a confirmed consensus (not a blind one)
        winning = winning if winning != "Adjudicator Override" else "Confirmed Consensus"

    return {
        "final_diagnosis":  str(result.get("final_diagnosis", result.get("diagnosis", cat))),
        "mapped_category":  str(cat),
        "confidence_score": int(result.get("confidence_score", 50)),
        "final_reasoning":  str(result.get("final_reasoning", result.get("primary_reasoning", "No reasoning provided."))),
        "winning_agent":    str(winning),
    }
