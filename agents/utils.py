"""
agents/utils.py
---------------
Shared utilities for parsing LLM responses, especially for reasoning models (R1/OpenThinker).
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
    Scan the text for any of the 21 PHMRC categories.
    Only used if JSON parsing fails completely.
    """
    # 1. Look for explicit [FINAL_DIAGNOSIS] tags first
    tag_match = re.search(r"\[FINAL_DIAGNOSIS\]\s*(.*?)\s*\[/FINAL_DIAGNOSIS\]", text, re.IGNORECASE)
    if tag_match:
        tag_val = tag_match.group(1).strip()
        # Find closest match in categories
        for cat in PHMRC_CATEGORIES:
            if cat.lower() in tag_val.lower():
                return {
                    "diagnosis": cat,
                    "confidence": "High (Tag Match)",
                    "primary_reasoning": "Extracted from [FINAL_DIAGNOSIS] tags.",
                    "fallback": True
                }

    # 2. General keyword scan (greedy but length-ordered)
    text_lower = text.lower()
    for cat in PHMRC_CATEGORIES:
        if cat.lower() in text_lower:
            return {
                "diagnosis": cat,
                "confidence": "Medium (Keyword Match)",
                "primary_reasoning": f"JSON parsing failed, but '{cat}' was found in text.",
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
            return parsed
        except:
            pass

    # 3. Return absolute fallback (tags or keyword)
    return tags if tags else {}
