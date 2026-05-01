"""
Translation Engine for PHMRC Verbal Autopsy — Child Module
===========================================================
Converts structured survey data + caregiver narratives
into clinical prose "Patient Dossiers" for LLM agents.

Outputs:
    - patient_dossiers.json  (structured, for LLM agents)
    - patient_dossiers.csv   (flat, for quick inspection)

Usage:
    python translation_engine.py
"""

import json
import pandas as pd
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

CSV_PATH = "child_va_form.csv"
NARRATIVE_PATH = "child_narratives_only.csv"
NARRATIVE_MIN_LENGTH = 50

SITE_MAP = {
    "AP": "Andhra Pradesh, India",
    "Dar": "Dar es Salaam, Tanzania",
    "UP": "Uttar Pradesh, India",
    "Pemba": "Pemba, Tanzania",
    "Bohol": "Bohol, Philippines",
    "Mexico": "Mexico City, Mexico",
}

CARE_COLUMNS = {
    "c5_02_1": "Traditional healer",
    "c5_02_2": "Homeopath",
    "c5_02_3": "Religious leader",
    "c5_02_4": "Government hospital",
    "c5_02_5": "Government health center/clinic",
    "c5_02_6": "Private hospital",
    "c5_02_7": "Community-based practitioner",
    "c5_02_8": "Trained birth attendant",
    "c5_02_9": "Private physician",
    "c5_02_10": "Pharmacy",
    "c5_02_11a": "Other provider",
    "c5_02_12": "Relative/friend",
}

MATERNAL_COMPLICATIONS = {
    "c2_01_1": "maternal convulsions",
    "c2_01_2": "maternal hypertension",
    "c2_01_3": "maternal anemia",
    "c2_01_4": "maternal diabetes",
    "c2_01_5": "non-headfirst delivery",
    "c2_01_6": "cord delivered first",
    "c2_01_7": "cord around neck",
    "c2_01_8": "excessive bleeding",
    "c2_01_9": "fever during labor",
    "c2_01_14": "placenta came out first",
}

INJURY_TYPES = {
    "c4_47_1": "road traffic injury",
    "c4_47_2": "fall",
    "c4_47_3": "drowning",
    "c4_47_4": "poisoning",
    "c4_47_5": "bite/sting",
    "c4_47_6": "burn/fire",
    "c4_47_7": "violence",
    "c4_47_8a": "other injury",
}


# =============================================================================
# HELPERS
# =============================================================================

def safe_val(row, col):
    """Get value from row. Returns None if missing/NaN/DontKnow/Refused/0."""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str in ("", "Don't Know", "Refused to Answer", "Refused", "0", "0.0"):
        return None
    return val_str


def is_yes(row, col):
    val = safe_val(row, col)
    return val is not None and val.lower() == "yes"


def is_no(row, col):
    val = safe_val(row, col)
    return val is not None and val.lower() == "no"


def get_age_string(row):
    """Build age string from g1_07a (years), g1_07b (months), g1_07c (days)."""
    years = safe_val(row, "g1_07a")
    months = safe_val(row, "g1_07b")
    days = safe_val(row, "g1_07c")
    parts = []
    if years and years not in ("999", "999.0"):
        y = int(float(years))
        if y > 0:
            parts.append(f"{y} year{'s' if y != 1 else ''}")
    if months and months not in ("99", "99.0"):
        m = int(float(months))
        if m > 0:
            parts.append(f"{m} month{'s' if m != 1 else ''}")
    if days and days not in ("99", "99.0"):
        d = int(float(days))
        if d > 0:
            parts.append(f"{d} day{'s' if d != 1 else ''}")
    return " ".join(parts) if parts else "Unknown age"


def get_duration(row, col):
    """Try to parse a numeric duration value."""
    val = safe_val(row, col)
    if val is None:
        return None
    try:
        d = int(float(val))
        return d if d > 0 else None
    except ValueError:
        return None


# =============================================================================
# LAYER 1: DEMOGRAPHICS
# =============================================================================

def build_demographics(row):
    age = get_age_string(row)
    sex = safe_val(row, "g1_05") or "Unknown"
    site_code = safe_val(row, "site") or "Unknown"
    site_full = SITE_MAP.get(site_code, site_code)
    return f"Age: {age} | Sex: {sex} | Site: {site_full}"


# =============================================================================
# LAYER 2: BIRTH HISTORY
# =============================================================================

def build_birth_history(row):
    parts = []
    birth_type = safe_val(row, "c1_01")
    if birth_type:
        parts.append(f"Birth: {birth_type}")
    birth_order = safe_val(row, "c1_02")
    if birth_order and birth_type and birth_type.lower() == "multiple":
        parts.append(f"Birth order: {birth_order}")
    birthplace = safe_val(row, "c1_06a")
    if birthplace:
        parts.append(f"Delivery: {birthplace}")
    weight = safe_val(row, "c1_08b")
    if weight:
        try:
            w = float(weight)
            if w > 0:
                parts.append(f"Weight: {int(w)}g")
        except ValueError:
            pass
    size = safe_val(row, "c1_07")
    if size:
        parts.append(f"Size: {size}")
    mother_alive = safe_val(row, "c1_03")
    if mother_alive:
        parts.append(f"Mother alive: {mother_alive}")
    return " | ".join(parts) if parts else "No birth history available."


# =============================================================================
# LAYER 3: MATERNAL HISTORY
# =============================================================================

def build_maternal_history(row):
    findings = []
    comp_list = [desc for col, desc in MATERNAL_COMPLICATIONS.items() if is_yes(row, col)]
    if comp_list:
        findings.append(f"Complications: {', '.join(comp_list)}")
    elif is_yes(row, "c2_01_10"):
        findings.append("No pregnancy/delivery complications reported.")
    preg_timing = safe_val(row, "c2_03")
    if preg_timing:
        findings.append(f"Pregnancy ended: {preg_timing}")
    delivery_type = safe_val(row, "c2_17")
    if delivery_type:
        findings.append(f"Delivery type: {delivery_type}")
    who_delivered = safe_val(row, "c2_15a")
    if who_delivered:
        findings.append(f"Delivered by: {who_delivered}")
    return " | ".join(findings) if findings else "No maternal history available."


# =============================================================================
# LAYER 4: ILLNESS TIMELINE
# =============================================================================

def build_illness_timeline(row):
    parts = []
    onset = safe_val(row, "c1_20")
    if onset:
        parts.append(f"Onset age: {onset}")
    duration = safe_val(row, "c1_21")
    if duration:
        parts.append(f"Duration: {duration}")
    place = safe_val(row, "c1_22a")
    if place:
        parts.append(f"Place of death: {place}")
    dd = safe_val(row, "c1_24d")
    dm = safe_val(row, "c1_24m")
    dy = safe_val(row, "c1_24y")
    date_parts = [p for p in [dd, dm, dy] if p]
    if date_parts:
        parts.append(f"Date of death: {' '.join(date_parts)}")
    return " | ".join(parts) if parts else "No timeline information available."


# =============================================================================
# LAYER 5: CLINICAL PRESENTATION (SYMPTOMS BY MEDICAL SYSTEM)
# =============================================================================

def build_fever(row):
    if not is_yes(row, "c4_01"):
        return "No fever reported."
    parts = []
    severity = safe_val(row, "c4_04")
    if severity:
        parts.append(severity.lower())
    pattern = safe_val(row, "c4_05")
    if pattern:
        parts.append(pattern.lower())
    parts.append("fever")
    d = get_duration(row, "c4_02")
    if d:
        parts.append(f"lasting {d} day{'s' if d != 1 else ''}")
    if is_yes(row, "c4_03"):
        parts.append("(continued until death)")
    return " ".join(parts).capitalize() + "."


def build_respiratory(row):
    findings = []
    if is_yes(row, "c4_12"):
        txt = "Severe cough" if is_yes(row, "c4_14") else "Cough present"
        d = get_duration(row, "c4_13")
        if d:
            txt += f" lasting {d} day{'s' if d != 1 else ''}"
        if is_yes(row, "c4_15"):
            txt += " with vomiting after cough"
        findings.append(txt + ".")
    elif is_no(row, "c4_12"):
        findings.append("No cough reported.")
    if is_yes(row, "c4_16"):
        txt = "Difficult breathing"
        d = get_duration(row, "c4_17")
        if d:
            txt += f" lasting {d} day{'s' if d != 1 else ''}"
        findings.append(txt + ".")
    elif is_no(row, "c4_16"):
        findings.append("No difficult breathing reported.")
    if is_yes(row, "c4_18"):
        txt = "Fast breathing"
        d = get_duration(row, "c4_19")
        if d:
            txt += f" lasting {d} day{'s' if d != 1 else ''}"
        findings.append(txt + ".")
    elif is_no(row, "c4_18"):
        findings.append("No fast breathing reported.")
    if is_yes(row, "c4_20"):
        findings.append("Chest indrawing present.")
    elif is_no(row, "c4_20"):
        findings.append("No chest indrawing reported.")
    sounds = []
    if is_yes(row, "c4_22"): sounds.append("stridor")
    if is_yes(row, "c4_23"): sounds.append("grunting")
    if is_yes(row, "c4_24"): sounds.append("wheezing")
    if sounds:
        findings.append(f"Breathing sounds: {', '.join(sounds)}.")
    else:
        neg = []
        if is_no(row, "c4_22"): neg.append("stridor")
        if is_no(row, "c4_23"): neg.append("grunting")
        if is_no(row, "c4_24"): neg.append("wheezing")
        if neg:
            findings.append(f"No {', '.join(neg)} reported.")
    return " ".join(findings)


def build_gi(row):
    findings = []
    if is_yes(row, "c4_06"):
        txt = "Diarrhea/loose stools present"
        d = get_duration(row, "c4_07b")
        if d:
            txt += f" (max {d} stools/day)"
        d2 = get_duration(row, "c4_08")
        if d2:
            txt += f", started {d2} day{'s' if d2 != 1 else ''} before death"
        if is_yes(row, "c4_09"):
            txt += " (continued until death)"
        findings.append(txt + ".")
        if is_yes(row, "c4_11"):
            findings.append("Blood in stool.")
        elif is_no(row, "c4_11"):
            findings.append("No blood in stool.")
    elif is_no(row, "c4_06"):
        findings.append("No diarrhea reported.")
    if is_yes(row, "c4_40"): findings.append("Protruding belly.")
    if is_yes(row, "c4_41"): findings.append("Weight loss.")
    if is_yes(row, "c4_43"): findings.append("White patches in mouth (oral thrush).")
    return " ".join(findings)


def build_neuro(row):
    findings = []
    if is_yes(row, "c4_25"):
        findings.append("Convulsions/fits present.")
    elif is_no(row, "c4_25"):
        findings.append("No convulsions reported.")
    if is_yes(row, "c4_26"):
        onset = safe_val(row, "c4_27")
        if onset:
            findings.append(f"Unconscious (onset: {onset} before death).")
        else:
            findings.append("Unconscious.")
    elif is_no(row, "c4_26"):
        findings.append("No unconsciousness reported.")
    if is_yes(row, "c4_28"):
        findings.append("Stiff neck present.")
    elif is_no(row, "c4_28"):
        findings.append("No stiff neck reported.")
    if is_yes(row, "c4_29"):
        findings.append("Bulging fontanelle present.")
    elif is_no(row, "c4_29"):
        findings.append("No bulging fontanelle reported.")
    return " ".join(findings)


def build_skin(row):
    findings = []
    if is_yes(row, "c4_30"):
        txt = "Skin rash present"
        loc1 = safe_val(row, "c4_31_1")
        loc2 = safe_val(row, "c4_31_2")
        locs = [l for l in [loc1, loc2] if l]
        if locs:
            txt += f" (location: {', '.join(locs)})"
        start = safe_val(row, "c4_32")
        if start:
            txt += f" (started on: {start})"
        if is_yes(row, "c4_34"):
            txt += " with blisters"
        findings.append(txt + ".")
    elif is_no(row, "c4_30"):
        findings.append("No rash reported.")
    if is_yes(row, "c4_35"): findings.append("Limb paralysis.")
    if is_yes(row, "c4_36"): findings.append("Swollen legs/feet (edema).")
    if is_yes(row, "c4_38"): findings.append("Skin flaking.")
    if is_yes(row, "c4_39"): findings.append("Hair color changed to red/yellow.")
    if is_yes(row, "c4_42"): findings.append("Lumps in neck/armpit.")
    if is_yes(row, "c4_46"): findings.append("Skin bruising.")
    return " ".join(findings)


def build_bleeding(row):
    if is_yes(row, "c4_44"):
        loc = safe_val(row, "c4_45")
        return f"Bleeding present (from: {loc})." if loc else "Bleeding from unspecified site."
    elif is_no(row, "c4_44"):
        return "No bleeding reported."
    return ""


def build_injury(row):
    injury_list = [desc for col, desc in INJURY_TYPES.items() if is_yes(row, col)]
    if not injury_list:
        return "No injury reported."
    result = f"Injury type: {', '.join(injury_list)}."
    if is_yes(row, "c4_48"):
        result += " Intentionally inflicted."
    elif is_no(row, "c4_48"):
        result += " Not intentional."
    survival = safe_val(row, "c4_49")
    if survival:
        try:
            s = int(float(survival))
            if s > 0:
                result += f" Survived {s} day{'s' if s != 1 else ''} after injury."
        except ValueError:
            pass
    return result


# =============================================================================
# LAYER 6: CARE-SEEKING
# =============================================================================

def build_care(row):
    parts = []
    if is_yes(row, "c5_01"):
        providers = [desc for col, desc in CARE_COLUMNS.items() if is_yes(row, col)]
        if providers:
            parts.append(f"Care sought: {', '.join(providers)}")
        else:
            parts.append("Care sought outside home (provider unspecified)")
    elif is_no(row, "c5_01"):
        parts.append("No care sought outside home.")
    hiv_parts = []
    if is_yes(row, "c5_17"):
        hiv_parts.append("Mother tested for HIV")
        if is_yes(row, "c5_18"):
            hiv_parts.append("(POSITIVE)")
        elif is_no(row, "c5_18"):
            hiv_parts.append("(negative)")
    if is_yes(row, "c5_19"):
        hiv_parts.append("Mother told she had AIDS")
    if hiv_parts:
        parts.append(" ".join(hiv_parts))
    return " | ".join(parts) if parts else "No care-seeking information available."


# =============================================================================
# LAYER 7: CAREGIVER NARRATIVE
# =============================================================================

def get_narrative(row, narratives_df):
    """Attach narrative if exists and is long enough."""
    newid = row.get("newid")
    if newid is None:
        return None
    try:
        match = narratives_df[narratives_df["newid"] == float(newid)]
    except (ValueError, TypeError):
        match = narratives_df[narratives_df["newid"].astype(str) == str(newid)]
    if match.empty:
        return None
    narrative = match.iloc[0].get("open_response")
    if pd.isna(narrative):
        return None
    narrative = str(narrative).strip()
    if len(narrative) < NARRATIVE_MIN_LENGTH:
        return None
    return narrative


# =============================================================================
# ASSEMBLE DOSSIER — STRUCTURED (dict) + FLAT (string)
# =============================================================================

def translate_case_structured(row, narratives_df):
    """Translate one row into a structured dict for JSON output."""
    newid = row.get("newid", "?")
    narrative = get_narrative(row, narratives_df)

    return {
        "case_id": str(newid),
        "ground_truth": row.get("gs_text34", "Unknown"),
        "sections": {
            "demographics": build_demographics(row),
            "birth_history": build_birth_history(row),
            "maternal_history": build_maternal_history(row),
            "illness_timeline": build_illness_timeline(row),
            "clinical_presentation": {
                "fever": build_fever(row),
                "respiratory": build_respiratory(row),
                "gastrointestinal": build_gi(row),
                "neurological": build_neuro(row),
                "skin_nutritional": build_skin(row),
                "bleeding": build_bleeding(row),
                "injury": build_injury(row),
            },
            "care_seeking": build_care(row),
            "caregiver_narrative": narrative if narrative else "No narrative available.",
        },
        "has_narrative": narrative is not None,
    }


def translate_case_flat(case_dict):
    """Convert structured dict into flat clinical prose string."""
    s = case_dict["sections"]
    cp = s["clinical_presentation"]

    lines = []
    lines.append(f"========== PATIENT DOSSIER — Case #{case_dict['case_id']} ==========")
    lines.append("")
    lines.append("DEMOGRAPHICS")
    lines.append(s["demographics"])
    lines.append("")
    lines.append("BIRTH HISTORY")
    lines.append(s["birth_history"])
    lines.append("")
    lines.append("MATERNAL HISTORY")
    lines.append(s["maternal_history"])
    lines.append("")
    lines.append("ILLNESS TIMELINE")
    lines.append(s["illness_timeline"])
    lines.append("")
    lines.append("CLINICAL PRESENTATION")
    lines.append(f"  [Fever]: {cp['fever']}")
    lines.append(f"  [Respiratory]: {cp['respiratory']}")
    lines.append(f"  [Gastrointestinal]: {cp['gastrointestinal']}")
    lines.append(f"  [Neurological]: {cp['neurological']}")
    lines.append(f"  [Skin/Nutritional]: {cp['skin_nutritional']}")
    lines.append(f"  [Bleeding]: {cp['bleeding']}")
    lines.append(f"  [Injury/Accident]: {cp['injury']}")
    lines.append("")
    lines.append("CARE-SEEKING")
    lines.append(s["care_seeking"])
    lines.append("")
    lines.append("CAREGIVER NARRATIVE")
    if case_dict["has_narrative"]:
        lines.append(f'"{s["caregiver_narrative"]}"')
    else:
        lines.append("No narrative available.")
    lines.append("")
    lines.append("=" * 55)
    return "\n".join(lines)


def translate_all(csv_df, narratives_df):
    """Translate all cases. Returns list of structured dicts."""
    results = []
    for idx in range(len(csv_df)):
        row = csv_df.iloc[idx]
        case = translate_case_structured(row, narratives_df)
        case["full_dossier"] = translate_case_flat(case)
        results.append(case)
    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    csv_df = pd.read_csv(CSV_PATH, low_memory=False)
    narratives_df = pd.read_csv(NARRATIVE_PATH)
    print(f"Loaded {len(csv_df)} cases | {len(narratives_df)} narratives\n")

    # Translate everything
    all_cases = translate_all(csv_df, narratives_df)

    # ---- JSON output (structured, for LLM agents) ----
    with open("patient_dossiers.json", "w") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_cases)} dossiers to patient_dossiers.json")

    # ---- CSV output (flat, for quick inspection) ----
    csv_rows = []
    for case in all_cases:
        csv_rows.append({
            "case_id": case["case_id"],
            "ground_truth": case["ground_truth"],
            "has_narrative": case["has_narrative"],
            "patient_dossier": case["full_dossier"],
        })
    pd.DataFrame(csv_rows).to_csv("patient_dossiers.csv", index=False)
    print(f"Saved {len(csv_rows)} dossiers to patient_dossiers.csv")

    # ---- Preview 3 diverse cases ----
    narr_count = sum(1 for c in all_cases if c["has_narrative"])
    print(f"\nStats: {narr_count} cases with narrative | {len(all_cases) - narr_count} without\n")

    causes_seen = set()
    for case in all_cases:
        if case["ground_truth"] not in causes_seen and len(causes_seen) < 3:
            causes_seen.add(case["ground_truth"])
            print(case["full_dossier"])
            print(f"  >>> GROUND TRUTH: {case['ground_truth']}\n")