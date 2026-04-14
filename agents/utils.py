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

PHMRC_CATEGORY_GUIDE = """- Pneumonia: cough, fast or difficult breathing, chest indrawing, respiratory distress, fever, crackling breath sounds
- Sepsis: sudden rapid deterioration, high fever or abnormally low temperature, signs of multiple organ involvement, no clear focal infection site
- Meningitis: neck stiffness, bulging fontanelle, photophobia, seizures, high fever, altered consciousness, kernig/brudzinski signs
- Encephalitis: brain inflammation, seizures, altered or fluctuating consciousness, fever, NO neck stiffness (distinguishes from meningitis)
- Malaria: cyclical or spiking high fever, residence or travel in endemic area, anaemia, splenomegaly, rapid deterioration
- Diarrhea/Dysentery: watery or bloody stools, severe dehydration, sunken eyes, skin tenting, rapid weight loss, cramping
- Measles: maculopapular rash starting on face spreading downward, high fever, cough, conjunctivitis, Koplik spots inside mouth
- Hemorrhagic fever: bleeding from multiple sites (nose, gums, skin), high fever, rash, very rapid deterioration, no obvious trauma
- AIDS: chronic wasting, recurrent or unusual infections, oral thrush, persistent lymphadenopathy, failure to thrive over months
- Other Infectious Diseases: clear infectious cause confirmed but symptoms do not fit Pneumonia, Sepsis, Meningitis, Malaria, Measles, or Diarrhea
- Drowning: found near or in water, submersion event reported, water in airways, asphyxia
- Road Traffic: history of vehicle collision, blunt or penetrating trauma, injuries consistent with impact
- Falls: reported fall from height, blunt trauma injuries localized to impact sites
- Fires: burn injuries, smoke inhalation, fire or explosion reported in history
- Violent Death: injuries inconsistent with reported history, signs of physical abuse, assault reported
- Poisonings: history of toxic substance ingestion (accidental or intentional), vomiting or altered consciousness after ingestion
- Bite of Venomous Animal: witnessed or reported snake or scorpion bite, local envenomation signs, systemic toxicity
- Other Cardiovascular Diseases: cardiac signs (murmur, cyanosis, oedema) without infection, trauma, or respiratory primary cause
- Other Cancers: chronic progressive illness, palpable mass, unexplained weight loss, non-infectious, non-traumatic
- Other Digestive Diseases: abdominal pathology without diarrhea — obstruction, perforation, intussusception, liver disease
- Other Defined Causes of Child Deaths: etiology is clearly identified but does not fit any category above"""

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


def strip_thoughts(text: str) -> str:
    """
    Remove <thought>...</thought> blocks from the model output.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
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

def parse_best_json(raw: str) -> dict:
    """
    Grabs the best-looking JSON block from the text.
    Prioritizes [FINAL_DIAGNOSIS] tags if present.
    """
    text = strip_thoughts(raw)
    
    # 1. Try to extract from tags first for better accuracy
    tags = _keyword_fallback(text)
    
    # 2. Try JSON extraction
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        candidate = _fix_dirty_json(match.group(1))
        try:
            parsed = json.loads(candidate)
            # If tags were found and they disagree with JSON, 
            # tags might be safer if JSON was malformed elsewhere
            if tags and "diagnosis" not in parsed:
                parsed["diagnosis"] = tags["diagnosis"]
            # Apply fuzzy matching to whatever diagnosis was parsed
            if "diagnosis" in parsed and parsed["diagnosis"]:
                resolved = fuzzy_match_category(str(parsed["diagnosis"]))
                if resolved:
                    parsed["diagnosis"] = resolved
            return parsed
        except:
            pass

    # 3. Return absolute fallback (tags or keyword)
    return tags if tags else {}
