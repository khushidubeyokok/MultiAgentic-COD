"""
agents/utils.py
---------------
Shared utilities for parsing LLM responses, especially for reasoning models (R1/OpenThinker).
Includes fuzzy category matching via a comprehensive alias table.
"""

import json
import re

# Sorted by length (descending) to prevent sub-string matching issues
PHMRC_CATEGORIES = sorted([
    "Drowning", "Poisonings", "Other Cardiovascular Diseases", "AIDS",
    "Violent Death", "Malaria", "Other Cancers", "Measles", "Meningitis",
    "Encephalitis", "Diarrhea/Dysentery", "Other Defined Causes of Child Deaths",
    "Other Infectious Diseases", "Hemorrhagic fever", "Other Digestive Diseases",
    "Bite of Venomous Animal", "Fires", "Falls", "Sepsis", "Pneumonia",
    "Road Traffic",
], key=len, reverse=True)

PHMRC_CATEGORY_GUIDE = """- Pneumonia: ACUTE illness (hours to days), cough PLUS fast/difficult breathing as the PRIMARY complaint, chest indrawing, fever, crackling breath sounds.
  ⛔ CAUTION — DO NOT diagnose Pneumonia if: the illness is chronic (>2 weeks of progressive decline without acute respiratory crisis), OR there is a documented chronic underlying condition (HIV/AIDS, cancer, severe malnutrition) — consider AIDS, Other Cancers, or Other Defined Causes instead. A cough mentioned alongside another primary illness does NOT make this Pneumonia.

- Sepsis: SUDDEN rapid multi-organ deterioration, high or abnormally low temperature, no single clear focal infection site — the systemic response overwhelms any focal diagnosis.
  ⛔ CRITICAL WARNING — Do NOT default to Sepsis just because you see fever. If the patient is from sub-Saharan Africa (Tanzania, Uganda, Kenya, Nigeria, Malawi, Mozambique, Ethiopia, etc.) AND has fever without clear bacterial source → consider Malaria FIRST. Sepsis should be your answer only after you have ruled out Malaria, Pneumonia, Meningitis, and Diarrhea.

- Meningitis: STIFF NECK (nuchal rigidity) is the KEY sign. Stiff neck + fever + seizures/altered consciousness = Meningitis. Bulging fontanelle, photophobia, Kernig/Brudzinski signs confirm.
  ⛔ If stiff neck is NOT explicitly documented in the dossier, do NOT diagnose Meningitis. Seizures + fever WITHOUT stiff neck = Encephalitis, not Meningitis.

- Encephalitis: Fever + seizures + altered or fluctuating consciousness — but NO STIFF NECK. If stiff neck is present in the dossier, choose Meningitis instead.

- Malaria: Cyclical or spiking high fever, history of residence/travel in malaria-endemic region (sub-Saharan Africa, South/Southeast Asia), anaemia, splenomegaly; rapid deterioration in young child.
  ⚠️ KEY RULE: In sub-Saharan African cases with undifferentiated fever and no clear bacterial source, Malaria is statistically far more likely than Sepsis. If location is African and primary complaint is high fever → Malaria is your top candidate.

- Diarrhea/Dysentery: Watery or bloody stools as PRIMARY complaint, severe dehydration, sunken eyes, skin tenting, rapid weight loss, cramping.
  ⛔ If diarrhea is mentioned but is NOT the primary complaint, do not diagnose this. Also: if diarrhea co-occurs with CNS/respiratory signs, the CNS signs may be secondary dehydration complications — Diarrhea/Dysentery is still the primary diagnosis.

- Measles: Maculopapular RASH starting on face spreading downward, high fever, cough, conjunctivitis, Koplik spots inside mouth.
  ⛔ A rash MUST be explicitly documented. Fever + cough alone is NOT Measles. Without documented rash, choose Pneumonia or Other Infectious Diseases.

- Hemorrhagic fever: SPONTANEOUS BLEEDING from MULTIPLE SITES (nose, gums, skin, urine/stool) with high fever; very rapid deterioration; no obvious trauma or chronic illness.
  ⛔ Bleeding from ONE site only does NOT qualify. Fever alone does NOT qualify. Both multi-site bleeding AND fever must be present simultaneously.

- AIDS: KEY MARKERS — oral thrush, recurrent infections (especially pneumonia that keeps recurring), chronic diarrhea >1 month, failure to thrive over months, mother known HIV-positive, persistent lymphadenopathy.
  ⚠️ If ANY 2+ of these appear alongside respiratory symptoms, strongly consider AIDS over Pneumonia. AIDS is a chronic, wasting illness — not a sudden acute event.

- Other Infectious Diseases: Confirmed infectious cause (e.g. typhoid, TB, pertussis, measles complications) that does NOT fit Pneumonia, Sepsis, Meningitis, Malaria, Measles, Diarrhea, or AIDS. Clear pathogen or syndrome but misfit for specific categories.

- Drowning: Found in or near water, submersion event reported by witness, water in airways, asphyxia pattern.

- Road Traffic: History of vehicle collision, blunt/penetrating trauma from road accident, injuries consistent with impact forces.

- Falls: Child found injured after a fall, caregiver reports fall or collapse from height. Head injury common.
  ⚠️ Even if neurological symptoms follow the fall, the cause is the FALL — classify as Falls, not CNS disease.

- Fires: Burn injuries, smoke inhalation, fire or explosion reported in history.

- Violent Death: Physical evidence of assault or abuse — injuries inconsistent with reported history, signs of beating or blunt force, homicide, assault documented.

- Poisonings: History of toxic substance ingestion (accidental or intentional), or strongly suspected ingestion — vomiting/altered consciousness after exposure without other explanation.

- Bite of Venomous Animal: Witnessed or reported snake or scorpion bite, local envenomation signs (swelling, necrosis), systemic toxicity.

- Other Cardiovascular Diseases: Cardiac signs (murmur, cyanosis, oedema, arrhythmia) as the PRIMARY pathology — without infection, trauma, or respiratory disease being primary.

- Other Cancers: Chronic progressive illness over WEEKS TO MONTHS, palpable abdominal or lymph node mass, unexplained persistent weight loss with NO clear infectious fever pattern. Child was unwell long before death.
  ⚠️ Chronic cough + weight loss + no acute fever = consider Other Cancers, not Pneumonia.

- Other Digestive Diseases: Abdominal pathology WITHOUT diarrhea as the primary presentation. Includes bowel obstruction, intussusception (sudden colicky pain in infant), appendicitis, perforation, liver disease (jaundice), hepatitis.
  ⚠️ KEY DISTINGUISHER from Diarrhea/Dysentery: diarrhea is NOT the main complaint here. Abdominal pain, vomiting, jaundice without diarrhea = Other Digestive Diseases.

- Other Defined Causes of Child Deaths: Specific identifiable etiology (prematurity, birth asphyxia, congenital anomaly, neonatal condition) that is clearly documented but does not fit any category above."""

# ── Fuzzy alias table ─────────────────────────────────────────────────────────
# Maps common misspellings, abbreviations, and variations → canonical category name.
# All keys are lowercase for case-insensitive matching.
CATEGORY_ALIASES: dict[str, str] = {
    # Diarrhea/Dysentery
    "diarrhea":                          "Diarrhea/Dysentery",
    "diarrhoea":                         "Diarrhea/Dysentery",
    "diarrhea/dysentery":                "Diarrhea/Dysentery",
    "diarrhoea/dysentery":               "Diarrhea/Dysentery",
    "dysentery":                         "Diarrhea/Dysentery",
    "gastroenteritis":                   "Diarrhea/Dysentery",
    "acute diarrhoea":                   "Diarrhea/Dysentery",
    "acute diarrhea":                    "Diarrhea/Dysentery",
    "diarrheal disease":                 "Diarrhea/Dysentery",

    # Pneumonia
    "pneumonia":                         "Pneumonia",
    "pneumonitis":                       "Pneumonia",
    "lower respiratory tract infection": "Pneumonia",
    "lrti":                              "Pneumonia",
    "acute respiratory infection":       "Pneumonia",
    "ari":                               "Pneumonia",
    "bronchopneumonia":                  "Pneumonia",

    # Malaria
    "malaria":                           "Malaria",
    "cerebral malaria":                  "Malaria",
    "falciparum malaria":                "Malaria",
    "severe malaria":                    "Malaria",

    # Meningitis
    "meningitis":                        "Meningitis",
    "bacterial meningitis":              "Meningitis",
    "viral meningitis":                  "Meningitis",
    "meningococcal disease":             "Meningitis",

    # Encephalitis
    "encephalitis":                      "Encephalitis",
    "viral encephalitis":                "Encephalitis",
    "meningo-encephalitis":              "Encephalitis",
    "meningoencephalitis":               "Encephalitis",

    # Sepsis
    "sepsis":                            "Sepsis",
    "septicemia":                        "Sepsis",
    "septicaemia":                       "Sepsis",
    "bacteremia":                        "Sepsis",
    "bacteraemia":                       "Sepsis",
    "neonatal sepsis":                   "Sepsis",

    # Measles
    "measles":                           "Measles",
    "rubeola":                           "Measles",

    # AIDS
    "aids":                              "AIDS",
    "hiv/aids":                          "AIDS",
    "hiv":                               "AIDS",
    "hiv disease":                       "AIDS",

    # Hemorrhagic fever
    "hemorrhagic fever":                 "Hemorrhagic fever",
    "haemorrhagic fever":                "Hemorrhagic fever",
    "viral hemorrhagic fever":           "Hemorrhagic fever",
    "ebola":                             "Hemorrhagic fever",
    "dengue hemorrhagic fever":          "Hemorrhagic fever",
    "dengue":                            "Hemorrhagic fever",

    # Road Traffic
    "road traffic":                      "Road Traffic",
    "road traffic accident":             "Road Traffic",
    "road traffic injury":               "Road Traffic",
    "rta":                               "Road Traffic",
    "motor vehicle accident":            "Road Traffic",
    "mva":                               "Road Traffic",
    "vehicle accident":                  "Road Traffic",
    "traffic accident":                  "Road Traffic",
    "road traffic crash":                "Road Traffic",

    # Drowning
    "drowning":                          "Drowning",
    "near drowning":                     "Drowning",
    "submersion":                        "Drowning",

    # Falls
    "falls":                             "Falls",
    "fall":                              "Falls",
    "fall from height":                  "Falls",
    "accidental fall":                   "Falls",

    # Fires
    "fires":                             "Fires",
    "burns":                             "Fires",
    "fire":                              "Fires",
    "burn injury":                       "Fires",
    "smoke inhalation":                  "Fires",

    # Violent Death
    "violent death":                     "Violent Death",
    "homicide":                          "Violent Death",
    "assault":                           "Violent Death",
    "physical abuse":                    "Violent Death",
    "violence":                          "Violent Death",

    # Poisonings
    "poisonings":                        "Poisonings",
    "poisoning":                         "Poisonings",
    "intoxication":                      "Poisonings",
    "toxic ingestion":                   "Poisonings",
    "accidental poisoning":              "Poisonings",

    # Bite of Venomous Animal
    "bite of venomous animal":           "Bite of Venomous Animal",
    "snake bite":                        "Bite of Venomous Animal",
    "snakebite":                         "Bite of Venomous Animal",
    "scorpion sting":                    "Bite of Venomous Animal",
    "envenomation":                      "Bite of Venomous Animal",
    "animal bite":                       "Bite of Venomous Animal",

    # Other Cardiovascular Diseases
    "other cardiovascular diseases":     "Other Cardiovascular Diseases",
    "cardiovascular disease":            "Other Cardiovascular Diseases",
    "congenital heart disease":          "Other Cardiovascular Diseases",
    "heart disease":                     "Other Cardiovascular Diseases",
    "cardiac disease":                   "Other Cardiovascular Diseases",
    "cardiomyopathy":                    "Other Cardiovascular Diseases",

    # Other Cancers
    "other cancers":                     "Other Cancers",
    "cancer":                            "Other Cancers",
    "malignancy":                        "Other Cancers",
    "tumor":                             "Other Cancers",
    "tumour":                            "Other Cancers",
    "lymphoma":                          "Other Cancers",
    "leukemia":                          "Other Cancers",
    "leukaemia":                         "Other Cancers",

    # Other Digestive Diseases
    "other digestive diseases":          "Other Digestive Diseases",
    "digestive disease":                 "Other Digestive Diseases",
    "bowel obstruction":                 "Other Digestive Diseases",
    "intussusception":                   "Other Digestive Diseases",
    "liver disease":                     "Other Digestive Diseases",
    "hepatitis":                         "Other Digestive Diseases",

    # Other Infectious Diseases
    "other infectious diseases":         "Other Infectious Diseases",
    "infectious disease":                "Other Infectious Diseases",
    "typhoid":                           "Other Infectious Diseases",
    "typhoid fever":                     "Other Infectious Diseases",
    "tuberculosis":                      "Other Infectious Diseases",
    "tb":                                "Other Infectious Diseases",
    "pertussis":                         "Other Infectious Diseases",
    "whooping cough":                    "Other Infectious Diseases",

    # Other Defined Causes of Child Deaths
    "other defined causes of child deaths": "Other Defined Causes of Child Deaths",
    "other defined causes":              "Other Defined Causes of Child Deaths",
    "prematurity":                       "Other Defined Causes of Child Deaths",
    "birth asphyxia":                    "Other Defined Causes of Child Deaths",
    "neonatal":                          "Other Defined Causes of Child Deaths",
    "stillbirth":                        "Other Defined Causes of Child Deaths",
    "congenital anomaly":                "Other Defined Causes of Child Deaths",
    "congenital malformation":           "Other Defined Causes of Child Deaths",
}


def fuzzy_match_category(text: str) -> str | None:
    """
    Attempt to resolve a free-text category name to a canonical PHMRC category.

    Resolution order:
    1. Exact match against PHMRC_CATEGORIES
    2. Alias table lookup (case-insensitive)
    3. Substring scan of PHMRC_CATEGORIES (length-ordered to avoid sub-matches)
    4. Returns None if no match found
    """
    if not text:
        return None

    text_stripped = text.strip()

    # 1. Exact match
    if text_stripped in PHMRC_CATEGORIES:
        return text_stripped

    # 2. Case-insensitive exact match
    text_lower = text_stripped.lower()
    for cat in PHMRC_CATEGORIES:
        if cat.lower() == text_lower:
            return cat

    # 3. Alias table (full text match)
    if text_lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[text_lower]

    # 4. Check if any alias is contained within the text (for descriptive phrases)
    for alias, canonical in CATEGORY_ALIASES.items():
        if alias in text_lower:
            return canonical

    # 5. Substring scan against canonical categories
    for cat in PHMRC_CATEGORIES:
        if cat.lower() in text_lower:
            return cat

    return None


GEMMA4_THINKING_PREFIX = "<|think|>\n"

def strip_thoughts(text: str) -> str:
    """
    Remove <thought>...</thought> and <think>...</think> blocks from output.
    Also handles gemma4 channel thought formats.
    """
    if not text:
        return ""
    
    # 1. Format A: Deepseek/OpenThinker blocks
    cleaned = re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<(thought|think)>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 2. Format B: gemma4 closed (<|channel>thought\n[reasoning]<channel|>)
    cleaned = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 3. Format C: gemma4 unclosed (<|channel>thought\n[reasoning never closes])
    cleaned = re.sub(r"<\|channel>thought\n.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Final pass: Strip any remaining XML-style tags and strip whitespace
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    
    return cleaned.strip()

def _fix_dirty_json(s: str) -> str:
    """
    Heuristically fix common 'dirty' JSON issues from LLMs.
    """
    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```\s*", "", s)
    s = re.sub(r"'(\w+)':", r'"\1":', s)
    s = re.sub(r",\s*\}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    return s.strip()

def _keyword_fallback(text: str) -> dict:
    """
    Scan the text for any of the 21 PHMRC categories using fuzzy matching.
    Only used if JSON parsing fails completely.
    """
    # 1. Look for explicit [FINAL_DIAGNOSIS] tags first
    tag_match = re.search(r"\[FINAL_DIAGNOSIS\]\s*(.*?)\s*\[/FINAL_DIAGNOSIS\]", text, re.IGNORECASE)
    if tag_match:
        tag_val = tag_match.group(1).strip()
        matched = fuzzy_match_category(tag_val)
        if matched:
            return {
                "diagnosis": matched,
                "confidence": "High (Tag Match)",
                "primary_reasoning": "Extracted from [FINAL_DIAGNOSIS] tags.",
                "fallback": True
            }

    # 2. General fuzzy scan
    matched = fuzzy_match_category(text)
    if matched:
        return {
            "diagnosis": matched,
            "confidence": "Medium (Keyword Match)",
            "primary_reasoning": f"JSON parsing failed, but '{matched}' was identified in text.",
            "fallback": True
        }
    return {}

def check_consensus(state: dict) -> bool:
    """
    Check if all 3 agents reached the same diagnosis after normalization.
    """
    a1 = state.get("agent1_output", {}).get("diagnosis")
    a2 = state.get("agent2_output", {}).get("diagnosis")
    a3 = state.get("agent3_output", {}).get("diagnosis")
    
    if not a1 or not a2 or not a3:
        return False
        
    m1 = fuzzy_match_category(str(a1))
    m2 = fuzzy_match_category(str(a2))
    m3 = fuzzy_match_category(str(a3))
    
    if not m1 or not m2 or not m3:
        return False
        
    return m1 == m2 == m3

def parse_best_json(raw: str) -> dict:
    """
    Grabs the best-looking JSON block from the text.
    Sorts all candidate blocks by length descending and returns the first one 
    that parses correctly and contains a "diagnosis" key.
    """
    text = strip_thoughts(raw)
    
    # 1. Try JSON extraction — find all {...} blocks
    # We use a greedy regex for candidates and then find all spans
    # We want to catch nested structures as single blocks if possible
    candidates = []
    
    # Find all starting points of '{'
    starts = [m.start() for m in re.finditer(r"\{", text)]
    # Find all ending points of '}'
    ends = [m.start() for m in re.finditer(r"\}", text)]
    
    # Generate all possible balanced or unbalanced { } spans
    # We'll rely on the parser to tell us if they are valid JSON
    for s in starts:
        for e in ends:
            if e > s:
                candidates.append(text[s:e+1])
    
    # Sort candidates by length descending (longest first)
    candidates.sort(key=len, reverse=True)

    for cand in candidates:
        candidate = _fix_dirty_json(cand)
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                continue
            
            # List of keys that indicate a valid diagnostic JSON
            valid_keys = ["diagnosis", "broad_group", "final_diagnosis", "mapped_category", "recommended_diagnosis", "consensus_diagnosis"]
            found_key = next((k for k in valid_keys if k in parsed and parsed[k]), None)

            if found_key:
                # Apply fuzzy matching to the primary diagnosis field found
                target_val = str(parsed[found_key])
                # We normalize the most important ones
                if found_key in ["diagnosis", "final_diagnosis", "mapped_category", "recommended_diagnosis", "consensus_diagnosis"]:
                    resolved = fuzzy_match_category(target_val)
                    if resolved:
                        # Update the key we found (or 'diagnosis' if it's a specialist)
                        parsed[found_key] = resolved
                        # Ensure 'diagnosis' key exists as a common interface for specialist nodes
                        if "diagnosis" not in parsed:
                            parsed["diagnosis"] = resolved
                
                # Add raw_response for debugging
                parsed["raw_response"] = raw
                return parsed
        except:
            continue

    # 2. Final Fallback: Try keyword extraction if JSON fails/is missing diagnosis
    tags = _keyword_fallback(text)
    if tags:
        tags["raw_response"] = raw
        return tags
        
    return {"raw_response": raw}
