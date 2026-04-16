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
    If categories is provided, only include those.
    """
    if not library:
        return ""
    
    targets = categories if categories else PHMRC_CATEGORIES
    block = "### EXEMPLAR CASES FOR REFERENCE (EXHAUSTIVE) ###\n"
    block += "Below are examples of dossiers and their correct ground-truth categories. Use these to calibrate your diagnostic criteria.\n\n"
    
    count = 0
    for cat in targets:
        if cat in library:
            item = library[cat]
            block += f"-- EXAMPLE {count+1} --\n"
            block += f"DOSSIER: {item['dossier'][:1500]}...\n"
            block += f"CORRECT CATEGORY: {cat}\n\n"
            count += 1
            
    return block
