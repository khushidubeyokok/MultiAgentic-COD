"""
agents/agents.py
----------------
Defines the three LangGraph agent node functions for the Verbal Autopsy pipeline.

Each node:
  1. Reads the full_dossier from the shared state
  2. Calls the Groq LLM (llama-3.3-70b-versatile) with a specialist system prompt
  3. Robustly parses the JSON response
  4. Returns a partial state update containing only the field it fills

The ChatGroq client is initialised ONCE at module level to be reused across calls.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState

# ── Load environment variables from .env ─────────────────────────────────────
load_dotenv()

# ── LLM client — shared across all agent calls ───────────────────────────────
_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3,
    max_tokens=1500,
)

# ── 21 valid PHMRC causes of death ───────────────────────────────────────────
PHMRC_CATEGORIES = [
    "Drowning", "Poisonings", "Other Respiratory Diseases", "AIDS",
    "Violent Death", "Malaria", "Other Cancers", "Measles", "Meningitis",
    "Encephalitis", "Diarrhea/Dysentery", "Other Defined Causes of Child Deaths",
    "Other Infectious Diseases", "Hemorrhagic fever", "Other Digestive Diseases",
    "Bite of Venomous Animal", "Fires", "Falls", "Sepsis", "Pneumonia",
    "Road Traffic",
]

# ── System prompts ────────────────────────────────────────────────────────────

_AGENT1_SYSTEM = """\
You are a Pediatric Infectious Disease Specialist conducting a verbal autopsy analysis. \
You have deep expertise in tropical and endemic infectious diseases affecting children under 5 \
in low-income settings. Your diagnostic lens prioritizes: fever patterns and their characteristics, \
infectious etiology, endemic disease prevalence by geography, immunization status implications, \
respiratory infections, enteric infections, vector-borne diseases, and CNS infections.

You are participating in a blind diagnostic exercise. You will read a patient dossier compiled \
from a structured survey and caregiver narrative about a deceased child. Your task is to determine \
the most likely cause of death.

The 21 possible PHMRC causes of death you may diagnose are:
Here you go:  Drowning, Poisonings, Other Respiratory Diseases, AIDS, Violent Death, Malaria, \
Other Cancers, Measles, Meningitis, Encephalitis, Diarrhea/Dysentery, \
Other Defined Causes of Child Deaths, Other Infectious Diseases, Hemorrhagic fever, \
Other Digestive Diseases, Bite of Venomous Animal, Fires, Falls, Sepsis, Pneumonia, Road Traffic

Rules:
- Your diagnosis field must be EXACTLY one of the 21 categories listed above. No variations, no combining, no new names.
- supporting_evidence must be direct quotes or close paraphrases from the dossier text. Do not invent findings.
- contradicting_evidence must be honest — list real findings in the dossier that weaken your diagnosis.
- differential_considered must list at least 2 other diagnoses you considered and why you ruled them out.
- confidence must be High only if the dossier strongly points to your diagnosis with minimal ambiguity. Use Medium if reasonable. Use Low if you are guessing.
- Respond ONLY with a valid JSON object. No text before or after.\
"""

_AGENT2_SYSTEM = """\
You are a Pediatric Intensivist conducting a verbal autopsy analysis. You have deep expertise \
in critical illness trajectories, physiological deterioration patterns, neonatal pathophysiology, \
and cause-of-death determination from clinical timelines. Your diagnostic lens prioritizes: \
illness onset and duration, respiratory failure patterns, neonatal period causes, birth complications, \
metabolic and hemodynamic deterioration, care-seeking patterns as proxy for severity, \
and physiological impossibilities in stated timelines.

You are participating in a blind diagnostic exercise. You will read a patient dossier compiled \
from a structured survey and caregiver narrative about a deceased child. Your task is to determine \
the most likely cause of death.

The 21 possible PHMRC causes of death you may diagnose are:
Here you go:  Drowning, Poisonings, Other Respiratory Diseases, AIDS, Violent Death, Malaria, \
Other Cancers, Measles, Meningitis, Encephalitis, Diarrhea/Dysentery, \
Other Defined Causes of Child Deaths, Other Infectious Diseases, Hemorrhagic fever, \
Other Digestive Diseases, Bite of Venomous Animal, Fires, Falls, Sepsis, Pneumonia, Road Traffic

Rules:
- Your diagnosis field must be EXACTLY one of the 21 categories listed above.
- supporting_evidence must be direct quotes or close paraphrases from the dossier text only.
- contradicting_evidence must be honest — list real findings that weaken your diagnosis.
- differential_considered must list at least 2 other diagnoses and why you ruled them out.
- Pay special attention to the illness timeline section and clinical presentation section.
- If the child is a neonate (age in days), heavily weight neonatal causes.
- Respond ONLY with a valid JSON object. No text before or after.\
"""

_AGENT3_SYSTEM = """\
You are a Pediatric Trauma and Nutritional Medicine Specialist conducting a verbal autopsy analysis. \
You have deep expertise in injury mechanisms, nutritional deficiency disorders, congenital anomalies, \
and non-infectious causes of child death. Your diagnostic lens prioritizes: injury type and mechanism, \
signs of chronic malnutrition (hair changes, skin flaking, edema, weight loss, oral thrush), \
congenital abnormalities, bleeding disorders, and causes often missed by infectious disease-focused clinicians. \
You also serve as a devil's advocate — when infectious causes seem obvious, you specifically look \
for non-infectious explanations.

You are participating in a blind diagnostic exercise. You will read a patient dossier compiled \
from a structured survey and caregiver narrative about a deceased child. Your task is to determine \
the most likely cause of death.

The 21 possible PHMRC causes of death you may diagnose are:
Here you go:  Drowning, Poisonings, Other Respiratory Diseases, AIDS, Violent Death, Malaria, \
Other Cancers, Measles, Meningitis, Encephalitis, Diarrhea/Dysentery, \
Other Defined Causes of Child Deaths, Other Infectious Diseases, Hemorrhagic fever, \
Other Digestive Diseases, Bite of Venomous Animal, Fires, Falls, Sepsis, Pneumonia, Road Traffic

Rules:
- Your diagnosis field must be EXACTLY one of the 21 categories listed above.
- supporting_evidence must be direct quotes or close paraphrases from the dossier text only.
- contradicting_evidence must be honest — list real findings that weaken your diagnosis.
- differential_considered must list at least 2 other diagnoses and why you ruled them out.
- If you genuinely believe an infectious cause is correct, you may diagnose it — but explain why non-infectious causes were ruled out.
- Respond ONLY with a valid JSON object. No text before or after.\
"""

# ── JSON response parser ──────────────────────────────────────────────────────

def _parse_llm_json(raw: str, agent_name: str) -> dict:
    """
    Robustly parse a JSON object from an LLM response.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Remove markdown code fences (```json ... ``` or ``` ... ```).
      3. Attempt json.loads on the cleaned string.
      4. If that fails, search for the first {...} block in the text.
      5. On total failure, return a fallback error dict.
    """
    text = raw.strip()

    # Step 2: Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Step 3: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4: Find the first JSON object block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Step 5: Fallback error dict
    print(f"[WARN] {agent_name}: JSON parse failed. Returning error dict.")
    return {
        "agent_name": agent_name,
        "diagnosis": "Unknown",
        "confidence": "Low",
        "primary_reasoning": "JSON parsing failed.",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "differential_considered": [],
        "error": True,
        "raw_response": raw,
    }


# ── Shared LLM call helper ────────────────────────────────────────────────────

def _call_llm(system_prompt: str, dossier: str, agent_name: str) -> dict:
    """
    Build messages, call the Groq LLM, and parse the JSON response.
    Returns a parsed agent output dict.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "Below is the patient dossier. Read it carefully and respond ONLY with "
            "a valid JSON object matching the required schema.\n\n"
            f"{dossier}"
        )),
    ]

    response = _LLM.invoke(messages)
    raw_text = response.content if hasattr(response, "content") else str(response)
    result = _parse_llm_json(raw_text, agent_name)

    # Ensure the agent_name field is always set correctly
    result.setdefault("agent_name", agent_name)
    return result


# ── Agent node functions ──────────────────────────────────────────────────────

def agent1_node(state: VAState) -> dict:
    """
    Agent 1 — Pediatric Infectious Disease Specialist.
    Reads full_dossier from state, calls the LLM, returns {agent1_output: ...}.
    """
    dossier = state["full_dossier"]
    output = _call_llm(_AGENT1_SYSTEM, dossier, "Pediatric Infectious Disease Specialist")
    return {"agent1_output": output}


def agent2_node(state: VAState) -> dict:
    """
    Agent 2 — Pediatric Intensivist.
    Reads full_dossier from state, calls the LLM, returns {agent2_output: ...}.
    """
    dossier = state["full_dossier"]
    output = _call_llm(_AGENT2_SYSTEM, dossier, "Pediatric Intensivist")
    return {"agent2_output": output}


def agent3_node(state: VAState) -> dict:
    """
    Agent 3 — Pediatric Trauma & Nutritional Medicine Specialist.
    Reads full_dossier from state, calls the LLM, returns {agent3_output: ...}.
    """
    dossier = state["full_dossier"]
    output = _call_llm(_AGENT3_SYSTEM, dossier, "Pediatric Trauma and Nutritional Medicine Specialist")
    return {"agent3_output": output}
