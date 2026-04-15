"""
agents/preprocessor.py
----------------------
Cleans raw patient dossiers before they reach the specialist agents.

Steps:
  1. Strip negative findings (lines saying symptom was ABSENT).
  2. Count positive evidence per clinical domain and rank them.
  3. Prepend a ranked domain summary and mark likely secondary symptoms.

Usage:
    from agents.preprocessor import preprocess_dossier
    clean_text = preprocess_dossier(raw_dossier_text)
"""

import re

# ── Negative-finding patterns ──────────────────────────────────────────────────
# Lines matching these are stripped (they tell agents what is NOT present,
# which adds noise and distracts from what IS present).
_NEGATIVE_PATTERNS = [
    re.compile(r"\bno\b[\w\s,]*\b(cough|fever|rash|diarrhea|diarrhoea|bleeding|vomit|seizure|convulsion|stiff neck|jaundice|oedema|edema)\b", re.IGNORECASE),
    re.compile(r"\b(denies|denied|absent|absence of|without|not present|not reported|no history of|not observed|unremarkable)\b", re.IGNORECASE),
    re.compile(r"\b(negative for|ruled out|not noted|not documented)\b", re.IGNORECASE),
]

# ── Domain keyword maps ───────────────────────────────────────────────────────
# Each domain maps to keywords that count as positive evidence for it.
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Respiratory": [
        "cough", "fast breathing", "difficult breathing", "respiratory",
        "chest indrawing", "wheeze", "stridor", "dyspnea", "tachypnea",
        "crackling", "pneumonia", "breath", "sputum",
    ],
    "Gastrointestinal": [
        "diarrhea", "diarrhoea", "vomit", "vomiting", "watery stool", "bloody stool",
        "dehydration", "sunken eyes", "skin turgor", "loose stool", "dysentery",
        "abdominal", "gastro", "nausea", "bowel",
    ],
    "CNS/Neurological": [
        "seizure", "convulsion", "altered consciousness", "unconscious",
        "stiff neck", "neck stiffness", "bulging fontanelle", "photophobia",
        "kernig", "brudzinski", "encephalop", "meningit", "brain",
        "focal deficit", "coma",
    ],
    "Infectious/Febrile": [
        "fever", "high temperature", "rigor", "chills", "malaria",
        "measles", "rash", "lymph", "sepsis", "infection", "pyrexia",
    ],
    "Hemorrhagic": [
        "bleeding", "haemorrhage", "hemorrhage", "blood", "petechiae",
        "purpura", "ecchymosis", "epistaxis", "haematuria",
    ],
    "Trauma/External": [
        "fall", "trauma", "injury", "fracture", "wound", "burn",
        "fire", "drowning", "road traffic", "accident", "vehicle",
        "bite", "sting", "poison", "toxic", "ingestion",
    ],
    "Cardiovascular": [
        "murmur", "cyanosis", "oedema", "edema", "cardiac", "heart",
        "pallor", "anaemia", "anemia", "shock", "hypotension",
    ],
    "Nutritional/Wasting": [
        "malnutrition", "wasting", "underweight", "kwashiorkor", "marasmus",
        "failure to thrive", "weight loss", "stunting", "mid-upper arm",
    ],
}


def _is_negative_line(line: str) -> bool:
    """Return True if this line primarily conveys the ABSENCE of a finding."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _NEGATIVE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _score_domains(text: str) -> list[tuple[str, int]]:
    """
    Count positive keyword hits per domain. Returns domains sorted
    descending by evidence count (ties broken alphabetically).
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[domain] = count
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _secondary_note(ranked: list[tuple[str, int]]) -> str:
    """
    If the top domain is GI and there are CNS/Resp signals too, add a note
    warning agents that CNS/Resp findings may be secondary complications of GI.
    Similarly for other common confounders.
    """
    if not ranked:
        return ""
    top_domain = ranked[0][0]
    domain_names = [d for d, _ in ranked]

    notes = []
    if top_domain == "Gastrointestinal" and any(
        d in domain_names for d in ("CNS/Neurological", "Respiratory")
    ):
        notes.append(
            "⚠️  SECONDARY SYMPTOM NOTE: Prominent GI evidence detected. "
            "Any CNS (seizures, altered consciousness) or Respiratory (fast breathing) "
            "signs co-occurring with severe diarrhea may be SECONDARY COMPLICATIONS of "
            "dehydration/electrolyte imbalance — NOT independent CNS or respiratory disease. "
            "Do not anchor on the secondary domain."
        )
    if top_domain == "CNS/Neurological" and "Infectious/Febrile" in domain_names:
        notes.append(
            "⚠️  SECONDARY SYMPTOM NOTE: CNS signs with high fever — consider whether "
            "meningitis vs encephalitis vs febrile seizure secondary to infection is primary."
        )
    if top_domain == "Respiratory" and "Infectious/Febrile" in domain_names:
        notes.append(
            "⚠️  SECONDARY SYMPTOM NOTE: Respiratory distress with fever — "
            "likely primary pneumonia; rule out sepsis with respiratory involvement."
        )
    return "\n".join(notes)


def preprocess_dossier(raw_text: str) -> str:
    """
    Main entry point. Annotates a raw dossier string.

    Returns a preprocessed dossier with:
    - A ranked domain evidence header prepended
    - A secondary symptom caution note (if applicable)
    - The FULL original dossier text preserved (negative findings are kept
      because "no stiff neck", "no rash" etc. are diagnostically critical)
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    # Step 1 — Score domains on the FULL, unmodified dossier text
    ranked = _score_domains(raw_text)

    # Step 2 — Build the header block
    header_parts = ["### PREPROCESSED DOSSIER SUMMARY ###"]
    if ranked:
        header_parts.append("Evidence domain ranking (strongest → weakest):")
        for i, (domain, count) in enumerate(ranked, 1):
            marker = "  PRIMARY →" if i == 1 else "           "
            header_parts.append(f"{marker} {i}. {domain} ({count} signal(s))")
    else:
        header_parts.append("(No strong domain signals detected — use full text)")

    sec_note = _secondary_note(ranked)
    if sec_note:
        header_parts.append("")
        header_parts.append(sec_note)

    header_parts.append("### END SUMMARY — FULL DOSSIER FOLLOWS ###\n")
    header = "\n".join(header_parts)

    return f"{header}\n{raw_text}"
