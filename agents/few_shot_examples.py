"""
agents/few_shot_examples.py
---------------------------
Dynamic few-shot example selection from the PHMRC dataset.

On startup:
  1. Loads the full dataset
  2. Picks 1 clear example per category (21 total) as the "example bank"
  3. Stores their case_ids so they can be excluded from evaluation

At runtime (per case):
  - Picks 2 examples from different categories relevant to the specialist
  - Formats them using the condensed dossier format
  - Never picks an example from the same category as... well, we don't know
    the answer, so we just pick diverse examples from the specialist's domain
"""

import json
import random
from pathlib import Path
from agents.preprocessor import condense_dossier
from agents.utils import PHMRC_CATEGORIES

# ── Which categories each specialist covers ──────────────────────────────────
SPECIALIST_CATEGORIES = {
    "specialist_id": [
        "Pneumonia", "Malaria", "Measles", "Meningitis",
        "Diarrhea/Dysentery", "Hemorrhagic fever", "AIDS",
        "Other Infectious Diseases",
    ],
    "specialist_cc": [
        "Pneumonia", "Sepsis", "Meningitis", "Encephalitis",
        "Other Cardiovascular Diseases",
        "Other Defined Causes of Child Deaths",
    ],
    "specialist_tn": [
        "Drowning", "Road Traffic", "Bite of Venomous Animal",
        "Falls", "Fires", "Violent Death", "Poisonings",
        "Other Cancers", "Other Digestive Diseases",
    ],
}

# ── Module-level state ───────────────────────────────────────────────────────
_example_bank: dict[str, dict] = {}   # category → case dict
_excluded_ids: set[str] = set()       # case_ids to exclude from eval
_initialized = False


def init_example_bank(data_path: str, seed: int = 99):
    """
    Build the example bank: 1 case per category.
    Prefers cases WITH a caregiver narrative and shorter dossiers (clearer signal).
    """
    global _example_bank, _excluded_ids, _initialized

    path = Path(data_path)
    with open(path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    rng = random.Random(seed)

    # Group by ground_truth
    by_category: dict[str, list] = {}
    for case in all_cases:
        cat = case.get("ground_truth", "")
        if cat in PHMRC_CATEGORIES:
            by_category.setdefault(cat, []).append(case)

    # Pick 1 best example per category
    for cat, cases in by_category.items():
        # Prefer cases with narrative (more signal)
        with_narr = [c for c in cases if c.get("has_narrative")]
        pool = with_narr if with_narr else cases

        # Pick a medium-length case (not too short, not too long)
        pool.sort(key=lambda c: len(c.get("full_dossier", "")))
        mid = len(pool) // 2
        # Take from the middle third for balanced length
        start = max(0, mid - len(pool) // 6)
        end = min(len(pool), mid + len(pool) // 6 + 1)
        candidate_pool = pool[start:end] if end > start else pool
        chosen = rng.choice(candidate_pool)

        _example_bank[cat] = chosen
        _excluded_ids.add(str(chosen.get("case_id", "")))

    _initialized = True
    print(f"[FEW-SHOT] Example bank built: {len(_example_bank)} categories, "
          f"{len(_excluded_ids)} cases excluded from eval")


def get_excluded_case_ids() -> set[str]:
    """Return case_ids that are in the example bank (exclude from eval)."""
    return _excluded_ids.copy()


def get_examples_for_agent(persona: str, n: int = 2) -> str:
    """
    Dynamically select n few-shot examples relevant to this specialist.
    Returns formatted string ready to inject into the prompt.
    """
    if not _initialized:
        return ""

    # Get categories this specialist covers
    relevant_cats = SPECIALIST_CATEGORIES.get(persona, PHMRC_CATEGORIES[:6])

    # Pick n examples from relevant categories
    available = [cat for cat in relevant_cats if cat in _example_bank]
    if not available:
        return ""

    # Shuffle and pick n
    rng = random.Random()  # different each run for variety
    rng.shuffle(available)
    selected_cats = available[:n]

    parts = []
    for i, cat in enumerate(selected_cats, 1):
        case = _example_bank[cat]
        condensed = condense_dossier(case)
        parts.append(
            f"### EXAMPLE {i} ###\n"
            f"{condensed}\n\n"
            f"### CORRECT OUTPUT ###\n"
            f'{{"diagnosis": "{cat}", "confidence": "High", '
            f'"primary_reasoning": "Based on clinical findings consistent with {cat}."}}'
        )

    return "\n\n".join(parts)
