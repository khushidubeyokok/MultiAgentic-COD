"""
agents/critic.py
----------------
Structured Vote-Counter and Conflict Resolver.

The critic receives three agent diagnoses + reasoning and outputs a strictly
structured JSON verdict — NOT vague adversarial commentary.

Consensus  → {"verdict": "consensus", ...}
Split      → {"verdict": "split", "strongest_agent": "agentX", ...}
"""

import json

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import strip_thoughts, parse_best_json

_LLM = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    num_ctx=8192,
)

_CRITIC_SYSTEM = (
    "You are a Clinical Arbitrator — a structured vote-counter and conflict resolver. "
    "You do NOT give opinions on the correct diagnosis. "
    "Your only job is to evaluate the internal consistency of each agent's stated reasoning "
    "against the evidence cited from the dossier, and produce a strict JSON verdict."
)

_CRITIC_TASK = """### YOUR TASK ###

You have received three specialist diagnoses from agents with DIFFERENT reasoning protocols:
- Agent 1 (Evidence Collector) reasoned bottom-up from symptoms to syndrome to category.
- Agent 2 (Eliminator) reasoned top-down by ruling out categories not supported by the dossier.
- Agent 3 (Timeline Analyst) reasoned temporally through the clinical narrative arc.

### CONSENSUS RULES (READ CAREFULLY) ###
⚠️  "consensus" verdict REQUIRES ALL THREE AGENTS TO AGREE on the exact same PHMRC category.
    2 out of 3 agreeing is NOT consensus — it is a SPLIT with a majority candidate.
    Under no circumstances should you report "consensus" if any agent gave a different diagnosis.

Step 1 — Check for TRUE consensus (all 3 agree):
If and ONLY IF all three agents agree on the SAME PHMRC category, output this JSON:
{
  "verdict": "consensus",
  "strongest_agent": "all",
  "consensus_diagnosis": "<category name>",
  "flag_for_review": false,
  "reason": "unanimous agreement — all three agents independently reached the same category"
}

Step 2 — If ANY agent disagrees (including 2-vs-1 splits), evaluate whose reasoning is MOST
INTERNALLY CONSISTENT with the evidence cited from the dossier. This includes cases where
2 agents agree and 1 disagrees.
Do NOT use your own opinion of the correct diagnosis.
Evaluate: (a) Does the agent's cited evidence actually appear in the dossier? (b) Does the
reasoning chain from evidence to syndrome to category hold together logically?
Output this JSON:
{
  "verdict": "split",
  "strongest_agent": "agent1_evidence_collector / agent2_eliminator / agent3_timeline_analyst",
  "recommended_diagnosis": "<category name of the most evidence-supported agent>",
  "flag_for_review": true or false,
  "reason": "<one sentence: which specific evidence best supports this agent's reasoning chain>"
}

Set flag_for_review to true if two or more agents give VERY DIFFERENT categories with
similarly strong reasoning (genuine clinical ambiguity).

OUTPUT ONLY THE JSON OBJECT. No preamble, no explanation, no markdown fences."""


def critic_node(state: VAState) -> dict:
    """
    Structured critic: evaluates the three agent outputs and produces a
    machine-readable JSON verdict for the adjudicator.
    """
    a1 = state["agent1_output"]
    a2 = state["agent2_output"]
    a3 = state["agent3_output"]

    def _fmt(d: dict) -> str:
        name = d.get("agent_name", "unknown")
        diag = d.get("diagnosis", "Unknown")
        reason = d.get("primary_reasoning", "No reasoning provided.")
        return f"  [{name}] Diagnosis: {diag}\n  Reasoning: {reason}"

    user_prompt = (
        f"### DOSSIER (Partial) ###\n{state['full_dossier'][:8000]}\n\n"
        f"### SPECIALIST DIAGNOSES ###\n"
        f"Agent 1 (Evidence Collector):\n{_fmt(a1)}\n\n"
        f"Agent 2 (Eliminator):\n{_fmt(a2)}\n\n"
        f"Agent 3 (Timeline Analyst):\n{_fmt(a3)}\n\n"
        f"{_CRITIC_TASK}"
    )

    response = _LLM.invoke([
        SystemMessage(content=_CRITIC_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    raw_text = response.content if hasattr(response, "content") else str(response)
    cleaned = strip_thoughts(raw_text)

    # Try to parse as JSON; fall back to raw string
    try:
        verdict = json.loads(cleaned)
        if isinstance(verdict, dict) and "verdict" in verdict:
            return {"critique": json.dumps(verdict)}
    except Exception:
        pass

    # Secondary attempt via parse_best_json
    parsed = parse_best_json(cleaned)
    if parsed and "verdict" in parsed:
        return {"critique": json.dumps(parsed)}

    # Absolute fallback — return raw text so adjudicator can handle it
    return {"critique": cleaned}
