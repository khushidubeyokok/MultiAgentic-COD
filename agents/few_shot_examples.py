"""
agents/few_shot_examples.py
---------------------------
Selects one exemplar dossier per PHMRC category for few-shot prompt grounding.
"""

import random
from agents.utils import PHMRC_CATEGORIES

def select_exemplars(all_cases: list, seed: int = 42) -> dict:
    """
    Returns a dictionary mapping PHMRC category -> full dossier text.
    Ensures 1 example per category is picked if available.
    """
    rng = random.Random(seed)
    # Shuffle so we don't always pick the first case
    shuffled = all_cases[:]
    rng.shuffle(shuffled)
    
    library = {}
    for cat in PHMRC_CATEGORIES:
        for case in shuffled:
            if case.get("ground_truth") == cat:
                library[cat] = {
                    "case_id": case.get("case_id"),
                    "dossier": case.get("full_dossier")
                }
                break
    return library

def format_few_shot_block(library: dict, categories: list = None) -> str:
    """
    Formats selected exemplars into a single string block.
    Caps dossiers at 400 chars and limits to 5 total exemplars.
    Picked randomly from the provided categories or all categories.
    """
    if not library:
        return ""
    
    available_cats = [c for c in (categories if categories else PHMRC_CATEGORIES) if c in library]
    
    # Pick up to 5 random categories from the available relevant ones
    selected_cats = random.sample(available_cats, min(5, len(available_cats)))
    
    block = "### EXEMPLAR CASES FOR REFERENCE ###\n"
    block += "Below are examples of dossiers and their correct ground-truth categories.\n\n"
    
    for i, cat in enumerate(selected_cats):
        item = library[cat]
        block += f"-- EXAMPLE {i+1} --\n"
        # Cap at 400 characters
        block += f"DOSSIER: {item['dossier'][:400]}...\n"
        block += f"CORRECT CATEGORY: {cat}\n\n"
            
    return block
