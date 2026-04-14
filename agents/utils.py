"""
agents/utils.py
---------------
Shared utilities: category constants, category guide, fuzzy matching, JSON parsing.
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

# Numbered list for prompts
PHMRC_NUMBERED_LIST = "\n".join(
    f"{i+1}. {cat}" for i, cat in enumerate(sorted(PHMRC_CATEGORIES))
)

# ── Category guide (from PHMRC clinical definitions) ─────────────────────────
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

# ── Common aliases/misspellings → canonical category ─────────────────────────
_CATEGORY_ALIASES = {
    "diarrhea": "Diarrhea/Dysentery",
    "dysentery": "Diarrhea/Dysentery",
    "diarrhoea": "Diarrhea/Dysentery",
    "diarrheal": "Diarrhea/Dysentery",
    "gastroenteritis": "Diarrhea/Dysentery",
    "snake bite": "Bite of Venomous Animal",
    "snakebite": "Bite of Venomous Animal",
    "venomous bite": "Bite of Venomous Animal",
    "scorpion": "Bite of Venomous Animal",
    "traffic accident": "Road Traffic",
    "car accident": "Road Traffic",
    "vehicle accident": "Road Traffic",
    "road accident": "Road Traffic",
    "rta": "Road Traffic",
    "burn": "Fires",
    "burns": "Fires",
    "fire": "Fires",
    "fall": "Falls",
    "drowning": "Drowning",
    "drowned": "Drowning",
    "poison": "Poisonings",
    "poisoning": "Poisonings",
    "malaria": "Malaria",
    "measles": "Measles",
    "meningitis": "Meningitis",
    "encephalitis": "Encephalitis",
    "pneumonia": "Pneumonia",
    "sepsis": "Sepsis",
    "septicemia": "Sepsis",
    "septicaemia": "Sepsis",
    "neonatal sepsis": "Sepsis",
    "aids": "AIDS",
    "hiv": "AIDS",
    "hemorrhagic": "Hemorrhagic fever",
    "haemorrhagic": "Hemorrhagic fever",
    "hemorrhagic fever": "Hemorrhagic fever",
    "dengue": "Hemorrhagic fever",
    "violent": "Violent Death",
    "homicide": "Violent Death",
    "assault": "Violent Death",
    "murder": "Violent Death",
    "cancer": "Other Cancers",
    "tumor": "Other Cancers",
    "tumour": "Other Cancers",
    "cardiovascular": "Other Cardiovascular Diseases",
    "heart": "Other Cardiovascular Diseases",
    "cardiac": "Other Cardiovascular Diseases",
}


def strip_thoughts(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _fix_dirty_json(s: str) -> str:
    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```\s*", "", s)
    s = re.sub(r"'(\w+)':", r'"\1":', s)
    s = re.sub(r",\s*\}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    return s.strip()


def fuzzy_match_category(text: str) -> str | None:
    if not text:
        return None
    text_clean = text.strip()
    if text_clean in PHMRC_CATEGORIES:
        return text_clean
    text_lower = text_clean.lower()
    for cat in PHMRC_CATEGORIES:
        if cat.lower() == text_lower:
            return cat
    for alias, cat in _CATEGORY_ALIASES.items():
        if alias in text_lower:
            return cat
    for cat in PHMRC_CATEGORIES:
        if cat.lower() in text_lower:
            return cat
    for cat in PHMRC_CATEGORIES:
        if text_lower in cat.lower() and len(text_lower) > 3:
            return cat
    return None


def _keyword_fallback(text: str) -> dict:
    tag_match = re.search(r"\[FINAL_DIAGNOSIS\]\s*(.*?)\s*\[/FINAL_DIAGNOSIS\]", text, re.IGNORECASE)
    if tag_match:
        tag_val = tag_match.group(1).strip()
        matched = fuzzy_match_category(tag_val)
        if matched:
            return {
                "diagnosis": matched,
                "confidence": "High",
                "primary_reasoning": "Extracted from [FINAL_DIAGNOSIS] tags.",
                "fallback": True
            }
    text_lower = text.lower()
    for cat in PHMRC_CATEGORIES:
        if cat.lower() in text_lower:
            return {
                "diagnosis": cat,
                "confidence": "Medium",
                "primary_reasoning": f"JSON parsing failed, but '{cat}' was found in text.",
                "fallback": True
            }
    for alias, cat in _CATEGORY_ALIASES.items():
        if alias in text_lower:
            return {
                "diagnosis": cat,
                "confidence": "Low",
                "primary_reasoning": f"JSON parsing failed, matched alias '{alias}' to '{cat}'.",
                "fallback": True
            }
    return {}


def parse_best_json(raw: str) -> dict:
    text = strip_thoughts(raw)
    tags = _keyword_fallback(text)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        candidate = _fix_dirty_json(match.group(1))
        try:
            parsed = json.loads(candidate)
            if "diagnosis" in parsed:
                matched = fuzzy_match_category(str(parsed["diagnosis"]))
                if matched:
                    parsed["diagnosis"] = matched
                elif tags and tags.get("diagnosis"):
                    parsed["diagnosis"] = tags["diagnosis"]
            elif tags and tags.get("diagnosis"):
                parsed["diagnosis"] = tags["diagnosis"]
            return parsed
        except Exception:
            pass
    return tags if tags else {}
