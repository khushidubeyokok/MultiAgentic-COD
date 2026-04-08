"""
agents/critic.py
----------------
Defines the critic_node LangGraph node function.

The Critic is an adversarial cross-examiner: it reads the three agent outputs
and the original dossier, then produces a plain-text critical analysis that
the Adjudicator will use to render the final verdict.

Returns: {"critique": <plain text string>}
"""

import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Shared LLM client (same model/settings as agents.py) ─────────────────────
_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3,
    max_tokens=1500,
)

# ── Critic system prompt ──────────────────────────────────────────────────────
_CRITIC_SYSTEM = """\
You are a Clinical Evidence Critic specializing in pediatric verbal autopsy quality assurance. \
You do not make diagnoses. Your sole job is adversarial cross-examination of three specialist diagnoses.

You will receive: the original patient dossier, and the structured diagnosis outputs from three \
specialists. You must rigorously challenge all three.

For each of the three agents, you must address:
1. HALLUCINATION CHECK: Did this agent cite any finding, symptom, or fact that does NOT appear in \
the original dossier? If yes, name it explicitly.
2. OMISSION CHECK: What significant finding in the dossier did this agent completely ignore that \
could change their diagnosis?
3. LOGICAL CONSISTENCY: Is the agent's reasoning internally consistent? Does their confidence level \
match the evidence they cited?
4. DIFFERENTIAL WEAKNESS: Is the diagnosis they ruled out actually stronger than the one they chose? \
Make the case.

Then add a CONVERGENCE SUMMARY: Do two or more agents agree? If yes, is the agreement based on the \
same evidence or coincidental? What is the single most unresolved question that the adjudicator must \
answer to make a final determination?

Be specific. Reference exact phrases from the dossier and exact claims from the agents. Do not be \
polite. Your job is to find weaknesses.\
"""


def _safe_list_str(value) -> str:
    """Convert a list or string to a readable string for the prompt."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value) if value else "None provided"
    return str(value) if value else "None provided"


def _build_user_prompt(state: VAState) -> str:
    """
    Populate the critic user prompt template with values from state.
    Gracefully handles agents that returned error dicts.
    """
    a1 = state["agent1_output"]
    a2 = state["agent2_output"]
    a3 = state["agent3_output"]

    def _get(d: dict, key: str, default: str = "Parse Error") -> str:
        return str(d.get(key, default)) if d else default

    return f"""ORIGINAL PATIENT DOSSIER:
{state["full_dossier"]}

AGENT 1 — PEDIATRIC INFECTIOUS DISEASE SPECIALIST:
Diagnosis: {_get(a1, "diagnosis")}
Confidence: {_get(a1, "confidence")}
Reasoning: {_get(a1, "primary_reasoning")}
Supporting Evidence: {_safe_list_str(a1.get("supporting_evidence", []))}
Contradicting Evidence: {_safe_list_str(a1.get("contradicting_evidence", []))}
Differentials Considered: {_safe_list_str(a1.get("differential_considered", []))}

AGENT 2 — PEDIATRIC INTENSIVIST:
Diagnosis: {_get(a2, "diagnosis")}
Confidence: {_get(a2, "confidence")}
Reasoning: {_get(a2, "primary_reasoning")}
Supporting Evidence: {_safe_list_str(a2.get("supporting_evidence", []))}
Contradicting Evidence: {_safe_list_str(a2.get("contradicting_evidence", []))}
Differentials Considered: {_safe_list_str(a2.get("differential_considered", []))}

AGENT 3 — PEDIATRIC TRAUMA AND NUTRITIONAL SPECIALIST:
Diagnosis: {_get(a3, "diagnosis")}
Confidence: {_get(a3, "confidence")}
Reasoning: {_get(a3, "primary_reasoning")}
Supporting Evidence: {_safe_list_str(a3.get("supporting_evidence", []))}
Contradicting Evidence: {_safe_list_str(a3.get("contradicting_evidence", []))}
Differentials Considered: {_safe_list_str(a3.get("differential_considered", []))}

Your task: Cross-examine all three diagnoses rigorously as described in your instructions."""


def critic_node(state: VAState) -> dict:
    """
    Critic node — adversarially challenges all three agent diagnoses.

    Reads:  full_dossier, agent1_output, agent2_output, agent3_output
    Returns: {"critique": <plain text critique string>}

    The output is intentionally plain text, not JSON.
    Rate-limit handling: on a 429, wait 60 s and retry once.
    """
    messages = [
        SystemMessage(content=_CRITIC_SYSTEM),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    def _call() -> str:
        response = _LLM.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    try:
        critique_text = _call()
    except Exception as exc:
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            print("[WARN] critic_node: Rate limit hit. Waiting 60 s before retry…")
            time.sleep(60)
            try:
                critique_text = _call()
            except Exception as retry_exc:
                print(f"[ERROR] critic_node: Retry failed — {retry_exc}")
                critique_text = f"Critique unavailable due to API error: {retry_exc}"
        else:
            raise

    return {"critique": critique_text}
