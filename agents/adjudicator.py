"""
agents/adjudicator.py
---------------------
Final adjudicator node.

Decision tiers (checked in order):
  1. Critic says consensus → return immediately (confidence 95)
  2. All 3 agent outputs agree → return immediately (confidence 95)
  3. 2/3 agents agree → majority vote, no LLM call (confidence 85)
  4. Critic recommends strongest agent → use that (confidence 75)
  5. Highest confidence agent wins programmatically (confidence 60)
  6. LLM fallback with critic hint (last resort)

Merged with: llm_config, fuzzy matching, programmatic voting.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import (
    parse_best_json, strip_thoughts, PHMRC_CATEGORIES,
    fuzzy_match_category, PHMRC_NUMBERED_LIST,
)
from agents.llm_config import get_llm

_LLM = get_llm(temperature=0.0)

PHMRC_LIST = ", ".join(PHMRC_CATEGORIES)

_ADJUDICATOR_SYSTEM = (
    "You are the Final Adjudicator — the last decision-maker in a structured multi-agent "
    "diagnostic panel. You have access to all specialist reasoning and a structured clinical "
    "critique. Your job is to render a single, authoritative final diagnosis."
)

_CONF_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def _get_confidence_weight(conf_str: str) -> int:
    if not conf_str:
        return 1
    first_word = str(conf_str).strip().split()[0].lower()
    return _CONF_WEIGHTS.get(first_word, 1)


def _get_valid_diagnosis(agent_output: dict) -> str | None:
    if not agent_output or agent_output.get("error"):
        return None
    diag = agent_output.get("diagnosis", "")
    return fuzzy_match_category(str(diag))


def _parse_critique(critique_str: str) -> dict | None:
    if not critique_str:
        return None
    try:
        obj = json.loads(critique_str)
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    except Exception:
        pass
    obj = parse_best_json(critique_str)
    if obj and "verdict" in obj:
        return obj
    return None


def _build_llm_prompt(state: VAState, critic_verdict: dict | None, candidates: list[str]) -> str:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]
    dossier = state.get("condensed_dossier") or state["full_dossier"][:8000]

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

    candidate_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    return f"""### PATIENT DOSSIER ###
{dossier}

### SPECIALIST INPUTS ###
Agent 1 (Evidence Collector): {_get(a1, "diagnosis")} | Reasoning: {_get(a1, "primary_reasoning")}
Agent 2 (Eliminator): {_get(a2, "diagnosis")} | Reasoning: {_get(a2, "primary_reasoning")}
Agent 3 (Timeline Analyst): {_get(a3, "diagnosis")} | Reasoning: {_get(a3, "primary_reasoning")}
{critic_hint}
### TASK ###
The specialists disagreed. Select the SINGLE most likely cause of death.

Choose from these candidates:
{candidate_list}

Or from the full list:
{PHMRC_NUMBERED_LIST}

⚠️ CONSTRAINT: Your mapped_category MUST be one of the diagnoses already proposed by the agents or the critic. Do NOT introduce a new category.

Reply with ONLY this JSON:
{{"mapped_category": "EXACT_CATEGORY", "confidence_score": 0-100, "final_reasoning": "one sentence", "winning_agent": "agent name"}}

[FINAL_DIAGNOSIS] Category [/FINAL_DIAGNOSIS]"""


def adjudicator_node(state: VAState) -> dict:
    a1, a2, a3 = state["agent1_output"], state["agent2_output"], state["agent3_output"]
    critique_str = state.get("critique", "")
    critic_verdict = _parse_critique(critique_str)

    d1 = _get_valid_diagnosis(a1)
    d2 = _get_valid_diagnosis(a2)
    d3 = _get_valid_diagnosis(a3)
    diagnoses = [d1, d2, d3]
    valid_diagnoses = [d for d in diagnoses if d is not None]
    agents = [a1, a2, a3]

    # ── TIER 1: Critic says consensus ────────────────────────────────────────
    if critic_verdict and critic_verdict.get("verdict") == "consensus":
        category = critic_verdict.get("consensus_diagnosis", "")
        matched = fuzzy_match_category(category)
        if matched:
            return {
                "final_diagnosis": matched,
                "mapped_category": matched,
                "confidence_score": 95,
                "final_reasoning": "Clinical critic confirmed unanimous consensus among all three agents.",
                "winning_agent": "Consensus (all agents)",
            }

    # ── TIER 2: All 3 agent outputs agree ────────────────────────────────────
    if d1 and d1 == d2 == d3:
        return {
            "final_diagnosis": str(d1),
            "mapped_category": str(d1),
            "confidence_score": 95,
            "final_reasoning": "Unanimous consensus among all three specialist agents.",
            "winning_agent": "Consensus (all agents)",
        }

    # ── TIER 3: Majority vote (2/3 agree) ────────────────────────────────────
    for candidate in valid_diagnoses:
        agreeing = [i for i, d in enumerate(diagnoses) if d == candidate]
        if len(agreeing) >= 2:
            conf_weights = [_get_confidence_weight(agents[i].get("confidence", "")) for i in agreeing]
            avg_conf = sum(conf_weights) / len(conf_weights)
            score = 85 if avg_conf >= 2.5 else 75 if avg_conf >= 2.0 else 65
            agent_names = [agents[i].get("agent_name", f"Agent {i+1}") for i in agreeing]
            return {
                "final_diagnosis": str(candidate),
                "mapped_category": str(candidate),
                "confidence_score": score,
                "final_reasoning": f"Majority vote: {', '.join(agent_names)} agreed on {candidate}.",
                "winning_agent": "Majority Vote",
            }

    # ── TIER 4: Highest confidence agent ────────────────────────────────────
    # Trust the agent that's most confident in its own answer first
    if valid_diagnoses:
        best_idx = -1
        best_weight = -1
        second_best_weight = -1
        for i, (agent, diag) in enumerate(zip(agents, diagnoses)):
            if diag is None:
                continue
            w = _get_confidence_weight(agent.get("confidence", ""))
            if w > best_weight:
                second_best_weight = best_weight
                best_weight = w
                best_idx = i
            elif w > second_best_weight:
                second_best_weight = w

        # Only use this tier if there's a clear confidence winner (not all tied)
        if best_idx >= 0 and best_weight > second_best_weight:
            chosen = diagnoses[best_idx]
            agent_name = agents[best_idx].get("agent_name", f"Agent {best_idx+1}")
            return {
                "final_diagnosis": str(chosen),
                "mapped_category": str(chosen),
                "confidence_score": 70,
                "final_reasoning": f"No consensus. Selected {agent_name}'s diagnosis (highest confidence).",
                "winning_agent": agent_name,
            }

    # ── TIER 5: Critic recommends strongest agent (tiebreaker) ───────────────
    # Only used when confidence scores are tied
    if critic_verdict and critic_verdict.get("verdict") == "split":
        recommended = critic_verdict.get("recommended_diagnosis", "")
        matched = fuzzy_match_category(recommended)
        if matched:
            strongest = critic_verdict.get("strongest_agent", "critic")
            reason = critic_verdict.get("reason", "")
            return {
                "final_diagnosis": matched,
                "mapped_category": matched,
                "confidence_score": 65,
                "final_reasoning": f"Critic tiebreaker — recommended {matched}: {reason}",
                "winning_agent": strongest,
            }

    # ── TIER 6: LLM fallback ────────────────────────────────────────────────
    candidates = list(set(valid_diagnoses)) if valid_diagnoses else PHMRC_CATEGORIES[:5]
    prompt = _build_llm_prompt(state, critic_verdict, candidates)
    response = _LLM.invoke([
        SystemMessage(content=_ADJUDICATOR_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)

    result = parse_best_json(raw_text)
    cat = result.get("mapped_category", result.get("diagnosis", ""))
    matched = fuzzy_match_category(str(cat))
    cat = matched or "Other Defined Causes of Child Deaths"

    return {
        "final_diagnosis": str(result.get("final_diagnosis", result.get("diagnosis", cat))),
        "mapped_category": str(cat),
        "confidence_score": int(result.get("confidence_score", 40)),
        "final_reasoning": str(result.get("final_reasoning", "LLM adjudication fallback.")),
        "winning_agent": str(result.get("winning_agent", "Adjudicator Override")),
    }
