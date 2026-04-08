"""
agents/adjudicator.py
---------------------
Defines the adjudicator_node LangGraph node function.

The Adjudicator reads all agent outputs + the Critic's analysis, and renders
the single final cause-of-death verdict as a structured JSON object.

Returns: {
    "final_diagnosis":  str,
    "mapped_category":  str,   # one of the 21 PHMRC categories
    "confidence_score": int,   # 0–100
    "final_reasoning":  str,
}
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import VAState

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Shared LLM client ─────────────────────────────────────────────────────────
_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3,
    max_tokens=1500,
)

# ── PHMRC category list (canonical, exact strings) ────────────────────────────
PHMRC_CATEGORIES = [
    "Drowning", "Poisonings", "Other Respiratory Diseases", "AIDS",
    "Violent Death", "Malaria", "Other Cancers", "Measles", "Meningitis",
    "Encephalitis", "Diarrhea/Dysentery", "Other Defined Causes of Child Deaths",
    "Other Infectious Diseases", "Hemorrhagic fever", "Other Digestive Diseases",
    "Bite of Venomous Animal", "Fires", "Falls", "Sepsis", "Pneumonia",
    "Road Traffic",
]

# ── Fuzzy-match lookup table: common LLM variants → canonical PHMRC name ─────
_FUZZY_LOOKUP: dict = {
    # Diarrhea variants
    "diarrhea":               "Diarrhea/Dysentery",
    "dysentery":              "Diarrhea/Dysentery",
    "diarrhoea":              "Diarrhea/Dysentery",
    "gastroenteritis":        "Diarrhea/Dysentery",
    # Respiratory variants
    "pneumonia":              "Pneumonia",
    "respiratory":            "Other Respiratory Diseases",
    "respiratory infection":  "Other Respiratory Diseases",
    "ari":                    "Other Respiratory Diseases",
    "acute respiratory":      "Other Respiratory Diseases",
    # Infection / sepsis
    "sepsis":                 "Sepsis",
    "septicemia":             "Sepsis",
    "bacteremia":             "Sepsis",
    "blood infection":        "Sepsis",
    "neonatal sepsis":        "Sepsis",
    # CNS
    "meningitis":             "Meningitis",
    "encephalitis":           "Encephalitis",
    "meningoencephalitis":    "Meningitis",
    "cerebral malaria":       "Malaria",
    # Infectious
    "malaria":                "Malaria",
    "measles":                "Measles",
    "hemorrhagic fever":      "Hemorrhagic fever",
    "haemorrhagic fever":     "Hemorrhagic fever",
    "viral hemorrhagic":      "Hemorrhagic fever",
    "hiv":                    "AIDS",
    "aids":                   "AIDS",
    # Digestive
    "digestive":              "Other Digestive Diseases",
    "intestinal obstruction": "Other Digestive Diseases",
    "intussusception":        "Other Digestive Diseases",
    # Injuries / external causes
    "drowning":               "Drowning",
    "poisoning":              "Poisonings",
    "poison":                 "Poisonings",
    "violent":                "Violent Death",
    "homicide":               "Violent Death",
    "assault":                "Violent Death",
    "fall":                   "Falls",
    "road traffic":           "Road Traffic",
    "road traffic accident":  "Road Traffic",
    "rta":                    "Road Traffic",
    "motor vehicle":          "Road Traffic",
    "fire":                   "Fires",
    "burn":                   "Fires",
    "snake":                  "Bite of Venomous Animal",
    "venomous":               "Bite of Venomous Animal",
    "bite":                   "Bite of Venomous Animal",
    # Cancer
    "cancer":                 "Other Cancers",
    "tumor":                  "Other Cancers",
    "tumour":                 "Other Cancers",
    "malignancy":             "Other Cancers",
    # Catch-alls
    "infectious":             "Other Infectious Diseases",
    "other infectious":       "Other Infectious Diseases",
    "congenital":             "Other Defined Causes of Child Deaths",
    "neonatal":               "Other Defined Causes of Child Deaths",
    "prematurity":            "Other Defined Causes of Child Deaths",
    "birth asphyxia":         "Other Defined Causes of Child Deaths",
    "sudden infant":          "Other Defined Causes of Child Deaths",
}


def _map_to_phmrc(raw_category: str) -> str:
    """
    Map a raw LLM-returned category string to the nearest PHMRC canonical name.

    Strategy:
      1. Exact match (case-insensitive, stripped).
      2. Substring match — canonical in raw or raw in canonical.
      3. Lookup table key scan.
      4. Fallback: "Other Defined Causes of Child Deaths" + warning.
    """
    if not raw_category:
        print("[WARN] adjudicator: empty mapped_category, using fallback.")
        return "Other Defined Causes of Child Deaths"

    cleaned = raw_category.strip()

    # Step 1: exact match
    for cat in PHMRC_CATEGORIES:
        if cat.lower() == cleaned.lower():
            return cat

    # Step 2: substring match
    cleaned_lower = cleaned.lower()
    for cat in PHMRC_CATEGORIES:
        if cat.lower() in cleaned_lower or cleaned_lower in cat.lower():
            print(f"[INFO] adjudicator: '{cleaned}' fuzzy-matched to '{cat}' (substring).")
            return cat

    # Step 3: lookup table
    for key, canonical in _FUZZY_LOOKUP.items():
        if key in cleaned_lower:
            print(f"[INFO] adjudicator: '{cleaned}' matched via lookup key '{key}' → '{canonical}'.")
            return canonical

    # Step 4: final fallback
    print(f"[WARN] adjudicator: Could not map '{cleaned}' to any PHMRC category. Using fallback.")
    return "Other Defined Causes of Child Deaths"


# ── Adjudicator system prompt ─────────────────────────────────────────────────
_ADJUDICATOR_SYSTEM = """\
You are the Clinical Adjudicator in a multi-specialist pediatric verbal autopsy panel. You have \
reviewed the original patient dossier, three specialist diagnoses, and an adversarial critique of \
all three. Your job is to render the single final cause of death verdict.

You are NOT a generalist. You are a meta-analyst. You do not introduce new diagnoses. You evaluate \
the evidence already presented and select the most defensible diagnosis.

Your decision criteria in order of priority:
1. Internal consistency: Which agent's reasoning holds up under the critic's scrutiny?
2. Evidence fidelity: Which diagnosis is most grounded in actual dossier findings, not assumptions?
3. Epidemiological plausibility: Given the child's age, site, symptoms, and illness duration, which \
cause is most plausible?
4. Convergence: If two or three agents agree, does the agreement survive the critic's challenge?
5. Hallucination penalty: Any agent found to have cited non-existent findings by the critic is penalized.

You must output a JSON object with exactly these fields:
{
  "final_diagnosis": "the cause of death in your own words",
  "mapped_category": "MUST be EXACTLY one of the 21 PHMRC categories listed below",
  "confidence_score": integer between 0 and 100,
  "final_reasoning": "2-3 paragraph explanation: which agent(s) you sided with and why, what the critic's challenge revealed, and why alternatives were rejected",
  "winning_agent": "Agent 1" or "Agent 2" or "Agent 3" or "Consensus" or "None - adjudicator override"
}

The 21 PHMRC categories you MUST choose from (use EXACTLY these strings, no variations):
 Drowning, Poisonings, Other Respiratory Diseases, AIDS, Violent Death, Malaria, Other Cancers, \
Measles, Meningitis, Encephalitis, Diarrhea/Dysentery, Other Defined Causes of Child Deaths, \
Other Infectious Diseases, Hemorrhagic fever, Other Digestive Diseases, Bite of Venomous Animal, \
Fires, Falls, Sepsis, Pneumonia, Road Traffic

Respond ONLY with a valid JSON object. No text before or after.\
"""


def _parse_llm_json(raw: str) -> dict:
    """Robustly parse a JSON object from the LLM response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print("[WARN] adjudicator: JSON parse failed. Returning error dict.")
    return {
        "final_diagnosis": "Parse Error",
        "mapped_category": "Other Defined Causes of Child Deaths",
        "confidence_score": 0,
        "final_reasoning": "Adjudicator JSON parse failed.",
        "winning_agent": "None - adjudicator override",
        "error": True,
        "raw_response": raw,
    }


def _build_user_prompt(state: VAState) -> str:
    """Build the adjudicator user prompt from current state."""
    a1 = state["agent1_output"]
    a2 = state["agent2_output"]
    a3 = state["agent3_output"]

    def _get(d: dict, key: str, default: str = "Parse Error") -> str:
        return str(d.get(key, default)) if d else default

    def _list_str(v) -> str:
        if isinstance(v, list):
            return "; ".join(str(i) for i in v) if v else "None provided"
        return str(v) if v else "None provided"

    return f"""ORIGINAL PATIENT DOSSIER (excerpt — use for grounding):
{state["full_dossier"][:3000]}
[...dossier continues if longer...]

AGENT 1 — PEDIATRIC INFECTIOUS DISEASE SPECIALIST:
Diagnosis: {_get(a1, "diagnosis")}  |  Confidence: {_get(a1, "confidence")}
Reasoning: {_get(a1, "primary_reasoning")}
Supporting: {_list_str(a1.get("supporting_evidence", []))}
Contradicting: {_list_str(a1.get("contradicting_evidence", []))}
Differentials: {_list_str(a1.get("differential_considered", []))}

AGENT 2 — PEDIATRIC INTENSIVIST:
Diagnosis: {_get(a2, "diagnosis")}  |  Confidence: {_get(a2, "confidence")}
Reasoning: {_get(a2, "primary_reasoning")}
Supporting: {_list_str(a2.get("supporting_evidence", []))}
Contradicting: {_list_str(a2.get("contradicting_evidence", []))}
Differentials: {_list_str(a2.get("differential_considered", []))}

AGENT 3 — PEDIATRIC TRAUMA AND NUTRITIONAL SPECIALIST:
Diagnosis: {_get(a3, "diagnosis")}  |  Confidence: {_get(a3, "confidence")}
Reasoning: {_get(a3, "primary_reasoning")}
Supporting: {_list_str(a3.get("supporting_evidence", []))}
Contradicting: {_list_str(a3.get("contradicting_evidence", []))}
Differentials: {_list_str(a3.get("differential_considered", []))}

CRITIC'S CROSS-EXAMINATION:
{state.get("critique", "No critique available.")}

Now render your final adjudication as a JSON object with the exact fields specified."""


def adjudicator_node(state: VAState) -> dict:
    """
    Adjudicator node — renders the final cause-of-death verdict.

    Reads:  full_dossier, agent1_output, agent2_output, agent3_output, critique
    Returns: {
        "final_diagnosis":  str,
        "mapped_category":  str,   # validated PHMRC category
        "confidence_score": int,
        "final_reasoning":  str,
    }
    """
    messages = [
        SystemMessage(content=_ADJUDICATOR_SYSTEM),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    def _call() -> str:
        response = _LLM.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    try:
        raw_text = _call()
    except Exception as exc:
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            print("[WARN] adjudicator_node: Rate limit hit. Waiting 60 s before retry…")
            time.sleep(60)
            try:
                raw_text = _call()
            except Exception as retry_exc:
                print(f"[ERROR] adjudicator_node: Retry failed — {retry_exc}")
                return {
                    "final_diagnosis": "API Error",
                    "mapped_category": "Other Defined Causes of Child Deaths",
                    "confidence_score": 0,
                    "final_reasoning": f"Adjudicator failed after retry: {retry_exc}",
                }
        else:
            raise

    result = _parse_llm_json(raw_text)

    # Validate and remap the mapped_category field
    raw_category = result.get("mapped_category", "")
    result["mapped_category"] = _map_to_phmrc(raw_category)

    result.setdefault("final_diagnosis", result.get("mapped_category", "Unknown"))
    result.setdefault("confidence_score", 0)
    result.setdefault("final_reasoning", "No reasoning provided.")

    return {
        "final_diagnosis":  str(result["final_diagnosis"]),
        "mapped_category":  str(result["mapped_category"]),
        "confidence_score": int(result.get("confidence_score", 0)),
        "final_reasoning":  str(result["final_reasoning"]),
    }
