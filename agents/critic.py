"""
agents/critic.py
----------------
Defines the critic_node LangGraph node function.
"""

import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState

_LLM = ChatOllama(
    model="openthinker:7b",
    temperature=0.0,
    num_ctx=8192,
)

def _strip_thoughts(text: str) -> str:
    """Strip <thought> blocks."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def critic_node(state: VAState) -> dict:
    """
    Critic node — provides an adversarial analysis of the three agent diagnoses.
    """
    a1 = state["agent1_output"]
    a2 = state["agent2_output"]
    a3 = state["agent3_output"]

    def _fmt(d: dict) -> str:
        return f"Diag: {d.get('diagnosis')} | Reason: {d.get('primary_reasoning')}"

    prompt = (
        "You are a Clinical Critic. Review the patient dossier and three specialist diagnoses. "
        "Identify inconsistencies or overlooked evidence.\n\n"
        f"### DOSSIER ###\n{state['full_dossier'][:3000]}\n\n"
        f"### DIAGNOSES ###\n"
        f"Agent 1: {_fmt(a1)}\n"
        f"Agent 2: {_fmt(a2)}\n"
        f"Agent 3: {_fmt(a3)}\n\n"
        "### TASK ###\n"
        "Provide a concise adversarial critique. Point out which agent's reasoning is strongest "
        "and where other agents might be hallucinating or missing key evidence. Output ONLY the critique text."
    )

    response = _LLM.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    
    return {"critique": _strip_thoughts(raw_text)}
