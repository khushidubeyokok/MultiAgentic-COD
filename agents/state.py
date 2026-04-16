"""
agents/state.py
---------------
Defines the shared LangGraph state for the Verbal Autopsy multi-agent pipeline.
All agent nodes read from and write to this TypedDict.
"""

from typing import TypedDict


class VAState(TypedDict):
    """
    Shared state that flows through the entire LangGraph pipeline.

    Fields populated by each stage:
      - case_id, ground_truth, has_narrative, full_dossier : set at graph entry
      - agent1_output, agent2_output, agent3_output        : filled by the three agent nodes
      - critique        : filled by the Critic node
      - final_diagnosis : filled by the Adjudicator node
      - mapped_category : one of the 21 PHMRC categories
      - confidence_score: 0–100 integer
      - final_reasoning : adjudicator's explanation
      - broad_group     : broad triage category ("External/Trauma", etc.)
    """

    # ── Triage / Broad Group (set by Stage 1) ───────────────────────────────
    broad_group: str      # "External/Trauma", "Infectious/Disease", or "Chronic/Systemic/Other"

    # ── Input fields (set before graph invocation) ──────────────────────────
    case_id: str          # e.g. "12345"
    ground_truth: str     # true cause of death — NEVER shown to agents
    has_narrative: bool   # whether a caregiver narrative is present
    full_dossier: str     # complete dossier text passed to every agent

    # ── Agent outputs (each is a dict matching the agent output schema) ─────
    # Standard Schema:
    # {
    #   "agent_name":             str,
    #   "diagnosis":              str,   # must be one of the 21 PHMRC categories
    #   "confidence":             str,   # exactly "High", "Medium", or "Low"
    #   "primary_reasoning":      str,
    #   # only present on parse failure:
    #   "error":                  bool,
    #   "raw_response":           str,
    # }
    # Note: Agent 2 (Symptom Scorer) adds 'scores' (dict) and 'top3' (list).
    agent1_output: dict   # Pediatric Infectious Disease Specialist
    agent2_output: dict   # Pediatric Intensivist
    agent3_output: dict   # Pediatric Trauma & Nutritional Medicine Specialist

    # ── Critic / Adjudicator fields ──────────────────────────────────────────
    critique: str         # filled by Critic node
    final_diagnosis: str  # filled by Adjudicator
    mapped_category: str  # one of the 21 PHMRC categories
    confidence_score: int # 0–100
    final_reasoning: str  # adjudicator's explanation
