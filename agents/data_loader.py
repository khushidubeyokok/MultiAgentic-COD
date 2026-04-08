"""
agents/data_loader.py
---------------------
Loads and optionally stratified-samples the patient_dossiers.json dataset.

Usage:
    from agents.data_loader import load_dossiers

    cases = load_dossiers("data/patient_dossiers.json", mode="demo", sample_size=30)
"""

import json
import math
import random
from pathlib import Path


# ── Field defaults applied when a dossier entry is missing keys ──────────────
_FIELD_DEFAULTS = {
    "case_id":      "",
    "ground_truth": "",
    "has_narrative": False,
    "full_dossier": "",
    "sections":     {},
}


def _apply_defaults(entry: dict) -> dict:
    """
    Ensure every required field exists in a dossier entry.
    Missing string fields default to empty string; has_narrative defaults to False.
    """
    normalised = {}
    for field, default in _FIELD_DEFAULTS.items():
        normalised[field] = entry.get(field, default)
    # Preserve any extra fields that may exist in the JSON
    for key in entry:
        if key not in normalised:
            normalised[key] = entry[key]
    return normalised


def _stratified_sample(
    cases: list,
    sample_size: int,
    rng: random.Random,
) -> list:
    """
    Stratified sampling by ground_truth category.

    Strategy:
      1. Compute floor(sample_size / n_categories) per category.
      2. Sample that many from each category (or all available if fewer).
      3. If total < sample_size, fill remaining slots randomly from the
         leftover pool (cases not already selected) using the same RNG.
    """
    # Group cases by category
    category_buckets: dict = {}
    for case in cases:
        cat = case["ground_truth"] or "UNKNOWN"
        category_buckets.setdefault(cat, []).append(case)

    n_categories = len(category_buckets)
    per_category = math.floor(sample_size / n_categories) if n_categories else 0

    selected: list = []
    leftover: list = []

    for cat, bucket in category_buckets.items():
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        take = min(per_category, len(shuffled))
        selected.extend(shuffled[:take])
        leftover.extend(shuffled[take:])

    # Fill remaining slots from the leftover pool
    remaining = sample_size - len(selected)
    if remaining > 0 and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:remaining])

    return selected


def load_dossiers(
    path,
    mode: str = "demo",
    sample_size: int = 30,
    seed: int = 42,
) -> list:
    """
    Load patient dossiers from a JSON file.

    Parameters
    ----------
    path        : path to patient_dossiers.json
    mode        : "demo" → stratified sample of sample_size cases
                  "full" → return all cases unchanged
    sample_size : target number of cases for demo mode (default 30)
    seed        : RNG seed for reproducibility (default 42)

    Returns
    -------
    List of dossier dicts, each with defaults applied for missing fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dossier file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Support both a plain list and a dict-wrapped list
    if isinstance(raw, dict):
        raw = list(raw.values())

    # Apply field defaults to every entry
    cases: list = [_apply_defaults(entry) for entry in raw]

    if mode == "full":
        result = cases
    elif mode == "demo":
        rng = random.Random(seed)
        result = _stratified_sample(cases, sample_size, rng)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'demo' or 'full'.")

    # ── Print summary ────────────────────────────────────────────────────────
    unique_cats = sorted({c["ground_truth"] or "UNKNOWN" for c in result})
    cat_counts: dict = {}
    for c in result:
        cat = c["ground_truth"] or "UNKNOWN"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Dossier Loader — mode={mode!r}")
    print(f"{'='*55}")
    print(f"  Total cases loaded  : {len(result)}")
    print(f"  Unique categories   : {len(unique_cats)}")
    print(f"  Category distribution:")
    for cat in sorted(cat_counts, key=lambda k: -cat_counts[k]):
        print(f"    {cat:<40} {cat_counts[cat]:>3}")
    print(f"{'='*55}\n")

    return result
