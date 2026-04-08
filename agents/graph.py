"""
agents/graph.py
---------------
Builds the LangGraph StateGraph for the Verbal Autopsy pipeline (Parts 1 + 2).

Final topology:
    START
      │
      ├──► agent1_node ──┐
      ├──► agent2_node ──┼──► critic_node ──► adjudicator_node ──► END
      └──► agent3_node ──┘

The three agent nodes run concurrently (fan-out). LangGraph merges their
partial state updates (fan-in at critic_node), then proceeds sequentially
through critic → adjudicator → END.
"""

import time
from langgraph.graph import StateGraph, START, END

from agents.state import VAState
from agents.agents import agent1_node, agent2_node, agent3_node
from agents.critic import critic_node
from agents.adjudicator import adjudicator_node

# ── Rate-limit retry wrapper ──────────────────────────────────────────────────

def _with_rate_limit_retry(node_fn):
    """
    Wraps an agent node function with a single rate-limit retry.

    If the Groq API returns a 429 on the first call, we wait 60 seconds
    and retry once. On a second failure we return an error-flagged dict
    for the field the node writes. critic_node and adjudicator_node handle
    their own retries internally.
    """
    def wrapper(state: VAState) -> dict:
        try:
            return node_fn(state)
        except Exception as exc:
            if "429" in str(exc) or "rate limit" in str(exc).lower():
                print(f"[WARN] Rate limit hit for {node_fn.__name__}. "
                      "Waiting 60 s before retry…")
                time.sleep(60)
                try:
                    return node_fn(state)
                except Exception as retry_exc:
                    print(f"[ERROR] Retry failed for {node_fn.__name__}: {retry_exc}")
                    error_dict = {
                        "agent_name": node_fn.__name__,
                        "diagnosis": "Unknown",
                        "confidence": "Low",
                        "primary_reasoning": "API call failed after retry.",
                        "supporting_evidence": [],
                        "contradicting_evidence": [],
                        "differential_considered": [],
                        "error": True,
                        "raw_response": str(retry_exc),
                    }
                    key_map = {
                        "agent1_node": "agent1_output",
                        "agent2_node": "agent2_output",
                        "agent3_node": "agent3_output",
                    }
                    field = key_map.get(node_fn.__name__, "agent1_output")
                    return {field: error_dict}
            raise

    wrapper.__name__ = node_fn.__name__
    return wrapper


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the full LangGraph StateGraph.

    Fan-out (parallel):  START → agent1_node, agent2_node, agent3_node
    Fan-in / join:       all three agents → critic_node
    Sequential:          critic_node → adjudicator_node → END

    LangGraph automatically waits for all incoming edges to a node before
    executing it, so critic_node acts as the implicit join point.

    Returns the compiled graph (ready for .invoke()).
    """
    builder = StateGraph(VAState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("agent1_node", _with_rate_limit_retry(agent1_node))
    builder.add_node("agent2_node", _with_rate_limit_retry(agent2_node))
    builder.add_node("agent3_node", _with_rate_limit_retry(agent3_node))
    builder.add_node("critic_node", critic_node)
    builder.add_node("adjudicator_node", adjudicator_node)

    # ── Fan-out: START → parallel agents ─────────────────────────────────────
    builder.add_edge(START, "agent1_node")
    builder.add_edge(START, "agent2_node")
    builder.add_edge(START, "agent3_node")

    # ── Fan-in: all agents → critic (implicit join — waits for all three) ────
    builder.add_edge("agent1_node", "critic_node")
    builder.add_edge("agent2_node", "critic_node")
    builder.add_edge("agent3_node", "critic_node")

    # ── Sequential: critic → adjudicator → END ────────────────────────────────
    builder.add_edge("critic_node", "adjudicator_node")
    builder.add_edge("adjudicator_node", END)

    return builder.compile()


# ── Compiled graph (module-level singleton) ───────────────────────────────────
graph = build_graph()


# ── Public entry point ────────────────────────────────────────────────────────

def run_single_case(case: dict) -> dict:
    """
    Run the full pipeline for a single patient case.

    Parameters
    ----------
    case : a single dossier dict (as returned by load_dossiers)

    Returns
    -------
    The final VAState dict after all nodes have completed.

    Notes
    -----
    - ground_truth is stored in state but NEVER passed to any agent prompt.
    """
    initial_state: VAState = {
        "case_id":       str(case.get("case_id", "")),
        "ground_truth":  str(case.get("ground_truth", "")),
        "has_narrative": bool(case.get("has_narrative", False)),
        "full_dossier":  str(case.get("full_dossier", "")),
        "agent1_output": {},
        "agent2_output": {},
        "agent3_output": {},
        "critique":         "",
        "final_diagnosis":  "",
        "mapped_category":  "",
        "confidence_score": 0,
        "final_reasoning":  "",
    }

    print(f"[INFO] Running pipeline for case_id={initial_state['case_id']} …")
    final_state = graph.invoke(initial_state)
    print(f"[INFO] Complete — case_id={initial_state['case_id']} | "
          f"verdict={final_state.get('mapped_category', 'N/A')}")
    return final_state
