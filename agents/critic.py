"""
agents/critic.py
----------------
Defines the critic_node LangGraph node function.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState
from agents.utils import strip_thoughts

_LLM = ChatOllama(
    model="openthinker:7b",
    temperature=0.1,  # Slight temperature for variety in adversarial voice
    num_ctx=8192,
)

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
        f"### DOSSIER (Partial) ###\n{state['full_dossier'][:5000]}\n\n"
        f"### DIAGNOSES ###\n"
        f"Agent 1: {_fmt(a1)}\n"
        f"Agent 2: {_fmt(a2)}\n"
        f"Agent 3: {_fmt(a3)}\n\n"
        "### TASK ###\n"
        "Provide a concise adversarial critique. Point out which agent's reasoning is strongest "
        "and where other agents might be hallucinating or missing key evidence. Output ONLY the critique text in short, not verbose."
    )

    response = _LLM.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    
    return {"critique": strip_thoughts(raw_text)}
