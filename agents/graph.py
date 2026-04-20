"""
agents/graph.py
---------------
Builds the LangGraph StateGraph for the Verbal Autopsy pipeline.

Topology:
    START
      │
    stage1_node (Triage)
      │
      ├──► agent1_node ──┐
      ├──► agent2_node ──┼──► join_agents (Logic) ──┬──► consensus_node ──► END
      └──► agent3_node ──┘                         │
                                                   └──► critic_node ──► adjudicator_node ──► END
"""

import time
from langgraph.graph import StateGraph, START, END

from agents.state import VAState
from agents.agents import agent1_node, agent2_node, agent3_node
from agents.adjudicator import adjudicator_node, consensus_node
from agents.stage1 import stage1_node
from agents.preprocessor import preprocess_dossier
from agents.utils import check_consensus

# ── Retry wrapper for transient API errors ───────────────────────────────────

def _with_retry(node_fn):
    key_map = {
        "agent1_node": "agent1_output",
        "agent2_node": "agent2_output",
        "agent3_node": "agent3_output",
    }

    def _error_dict(exc: Exception) -> dict:
        return {
            "agent_name": node_fn.__name__,
            "diagnosis": "Unknown",
            "confidence": "Low",
            "primary_reasoning": "API call failed after retry.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "differential_considered": [],
            "error": True,
            "raw_response": str(exc),
        }

    def wrapper(state: VAState) -> dict:
        try:
            return node_fn(state)
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit  = "429" in exc_str or "rate limit" in exc_str or "quota" in exc_str
            is_unavailable = "503" in exc_str or "unavailable" in exc_str or "overload" in exc_str

            if is_rate_limit:
                wait = 60
                label = "Rate limit (429)"
            elif is_unavailable:
                wait = 15
                label = "Server overload (503)"
            else:
                raise

            print(f"[WARN] {label} hit for {node_fn.__name__}. Waiting {wait} s before retry…")
            time.sleep(wait)

            try:
                return node_fn(state)
            except Exception as retry_exc:
                print(f"[ERROR] Retry failed for {node_fn.__name__}: {retry_exc}")
                field = key_map.get(node_fn.__name__, "agent1_output")
                return {field: _error_dict(retry_exc)}

    wrapper.__name__ = node_fn.__name__
    return wrapper


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(VAState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("stage1_node", stage1_node)
    builder.add_node("agent1_node", _with_retry(agent1_node))
    builder.add_node("agent2_node", _with_retry(agent2_node))
    builder.add_node("agent3_node", _with_retry(agent3_node))
    builder.add_node("adjudicator_node", adjudicator_node)
    builder.add_node("consensus_node", consensus_node)
    
    # Dummy join node to consolidate parallel agent outputs before branching
    def join_agents_node(state: VAState):
        return state
    builder.add_node("join_agents", join_agents_node)

    # ── Edges ────────────────────────────────────────────────────────────────
    builder.add_edge(START, "stage1_node")
    
    # Fan-out after Triage
    builder.add_edge("stage1_node", "agent1_node")
    builder.add_edge("stage1_node", "agent2_node")
    builder.add_edge("stage1_node", "agent3_node")

    # Fan-in: all agents → join_agents
    builder.add_edge("agent1_node", "join_agents")
    builder.add_edge("agent2_node", "join_agents")
    builder.add_edge("agent3_node", "join_agents")

    # Conditional branching: consensus vs split
    def route_after_agents(state: VAState):
        if check_consensus(state):
            return "consensus"
        return "split"

    builder.add_conditional_edges(
        "join_agents",
        route_after_agents,
        {
            "consensus": "consensus_node",
            "split": "adjudicator_node"
        }
    )

    # Path A: Split resolution (LLM Adjudication)
    builder.add_edge("adjudicator_node", END)

    # Path B: Consensus (Fast Path)
    builder.add_edge("consensus_node", END)

    return builder.compile()


# ── Compiled graph ───────────────────────────────────────────────────────────
graph = build_graph()


# ── Public entry point ────────────────────────────────────────────────────────

def run_single_case(case: dict) -> dict:
    raw_dossier = str(case.get("full_dossier", ""))
    preprocessed_dossier = preprocess_dossier(raw_dossier)

    initial_state: VAState = {
        "case_id":       str(case.get("case_id", "")),
        "ground_truth":  str(case.get("ground_truth", "")),
        "has_narrative": bool(case.get("has_narrative", False)),
        "full_dossier":  preprocessed_dossier,
        "broad_group":   "",
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
    
    final_state = initial_state
    for event in graph.stream(initial_state):
        for node_name, output in event.items():
            final_state.update(output)
            
            if "stage1" in node_name:
                print(f"  [STREAM] Triage Complete: Broad Group = {output.get('broad_group')}")
            elif "agent" in node_name and "join" not in node_name:
                field = node_name.replace("_node", "_output")
                diag = output.get(field, {}).get("diagnosis", "Unknown")
                conf = output.get(field, {}).get("confidence", "N/A")
                persona = node_name.replace("_", " ").title()
                print(f"  [STREAM] {persona} finished: {diag} [{conf}]")
            elif "adjudicator" in node_name:
                print(f"  [STREAM] Adjudicator final verdict rendered.")
            elif "consensus" in node_name:
                print(f"  [STREAM] Consensus fast-path triggered.")

    print(f"[INFO] Complete — case_id={initial_state['case_id']} | "
          f"verdict={final_state.get('mapped_category', 'N/A')}")
    return final_state
