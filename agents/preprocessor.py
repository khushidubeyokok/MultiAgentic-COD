"""
agents/preprocessor.py
----------------------
Programmatic (no LLM) extraction of key clinical features from patient dossiers.
Produces a condensed dossier that removes noise and keeps only positive findings.
Ranks symptom domains by number of findings to help identify primary cause.
"""


# Domain display names
_DOMAIN_LABELS = {
    "fever": "Fever",
    "respiratory": "Respiratory",
    "gastrointestinal": "Gastrointestinal",
    "neurological": "Neurological",
    "skin_nutritional": "Skin/Nutritional",
    "bleeding": "Bleeding",
    "injury_accident": "Injury/Accident",
    "injury": "Injury/Accident",
}

_SECONDARY_NOTES = {}  # Removed — let the model reason about symptom relationships


def condense_dossier(case: dict) -> str:
    """
    Extract positive clinical findings from structured sections.
    Ranks domains by finding count to identify the primary symptom domain.
    """
    sections = case.get("sections", {})
    if not sections:
        return case.get("full_dossier", "")[:2000]

    parts = []

    # Demographics + geographic alerts
    demo = sections.get("demographics", "")
    if demo:
        parts.append(f"DEMOGRAPHICS: {demo}")
        demo_lower = str(demo).lower()
        geo_notes = _get_geographic_alerts(demo_lower)
        if geo_notes:
            parts.append(geo_notes)

    # Illness timeline
    timeline = sections.get("illness_timeline", "")
    if timeline:
        parts.append(f"TIMELINE: {timeline}")

    # Clinical presentation — extract, clean, and RANK by finding count
    cp = sections.get("clinical_presentation", {})
    if isinstance(cp, dict):
        domain_findings = []  # list of (domain_key, label, cleaned_text, finding_count)

        for domain, finding in cp.items():
            finding_str = str(finding).strip()
            if not finding_str:
                continue
            if _is_negative_only(finding_str):
                continue
            cleaned = _extract_positive_sentences(finding_str)
            if not cleaned:
                continue

            label = _DOMAIN_LABELS.get(domain, domain.replace("_", "/").title())
            count = len([s for s in cleaned.split(".") if s.strip()])
            domain_findings.append((domain, label, cleaned, count))

        if domain_findings:
            # Sort by finding count descending
            domain_findings.sort(key=lambda x: -x[3])

            lines = []
            primary_domain = domain_findings[0][0] if domain_findings else None

            for i, (domain_key, label, text, count) in enumerate(domain_findings):
                if i == 0 and len(domain_findings) > 1:
                    lines.append(f"  ★ PRIMARY — {label} ({count} findings): {text}")
                else:
                    lines.append(f"  {label} ({count} findings): {text}")

            # Add secondary symptom notes if applicable
            # Check ALL domain pairs, not just primary→secondary
            if len(domain_findings) >= 2:
                domain_keys = [df[0] for df in domain_findings]
                added_notes = set()
                for key_pair, note in _SECONDARY_NOTES.items():
                    if key_pair[0] in domain_keys and key_pair[1] in domain_keys:
                        if note not in added_notes:
                            lines.append(f"  {note}")
                            added_notes.add(note)

            parts.append("CLINICAL FINDINGS (ranked by evidence):\n" + "\n".join(lines))

    elif isinstance(cp, str) and cp.strip():
        parts.append(f"CLINICAL: {cp}")

    # Care-seeking
    care = sections.get("care_seeking", "")
    if care and str(care).strip() and "No care" not in str(care):
        parts.append(f"CARE: {care}")

    # Caregiver narrative
    narrative = sections.get("caregiver_narrative", "")
    if narrative and str(narrative).strip():
        narr_str = str(narrative).strip()
        if len(narr_str) > 500:
            narr_str = narr_str[:500] + "..."
        parts.append(f"NARRATIVE: {narr_str}")

    return "\n".join(parts) if parts else case.get("full_dossier", "")[:2000]


def _get_geographic_alerts(demo_lower: str) -> str:
    """Generate disease alerts based on patient's geographic location."""
    alerts = []

    # Sub-Saharan Africa
    africa = ["tanzania", "nigeria", "kenya", "uganda", "mozambique",
              "dar es salaam", "pemba"]
    if any(r in demo_lower for r in africa):
        alerts.append("Malaria (highly endemic), AIDS (high prevalence), Measles (low vaccination areas)")

    # South Asia
    south_asia = ["india", "andhra pradesh", "uttar pradesh", "bangladesh", "myanmar"]
    if any(r in demo_lower for r in south_asia):
        alerts.append("Malaria (endemic), Dengue/Hemorrhagic fever (endemic), Measles (low vaccination areas)")

    # Southeast Asia
    southeast_asia = ["philippines", "bohol", "indonesia", "vietnam", "cambodia"]
    if any(r in demo_lower for r in southeast_asia):
        alerts.append("Dengue/Hemorrhagic fever (highly endemic), Malaria (endemic)")

    # Latin America
    latam = ["mexico", "brazil", "colombia", "peru"]
    if any(r in demo_lower for r in latam):
        alerts.append("Dengue/Hemorrhagic fever (endemic)")

    if alerts:
        return "⚠️ GEOGRAPHIC CONTEXT: Region is endemic for: " + "; ".join(alerts) + ". Consider these when fever is present."
    return ""


def _is_negative_sentence(s: str) -> bool:
    """Check if a single sentence is a negative finding."""
    s_lower = s.lower().strip()
    return (s_lower.startswith("no ") or
            "not reported" in s_lower or
            "not present" in s_lower or
            "not available" in s_lower)


def _is_negative_only(text: str) -> bool:
    """Check if a clinical finding contains only negative/absent findings."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return True
    return all(_is_negative_sentence(s) for s in sentences)


def _extract_positive_sentences(text: str) -> str:
    """Remove negative sentences, keep only positive clinical findings."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    positives = [s for s in sentences if not _is_negative_sentence(s)]
    return ". ".join(positives) + "." if positives else ""
