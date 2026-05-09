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

from agents.utils import PHMRC_CATEGORIES


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
    Proportional stratified sampling by ground_truth category.

    Uses the Largest Remainder Method (Hamilton method) to convert fractional
    proportional allocations into an exact integer sample size.
    """
    if sample_size <= 0 or not cases:
        return []

    if sample_size >= len(cases):
        selected = cases[:]
        rng.shuffle(selected)
        return selected

    category_buckets: dict = {}
    for case in cases:
        cat = case["ground_truth"] or "UNKNOWN"
        category_buckets.setdefault(cat, []).append(case)

    total_available = len(cases)
    floor_alloc: dict[str, int] = {}
    remainders: dict[str, float] = {}

    for cat, bucket in category_buckets.items():
        exact = len(bucket) * sample_size / total_available
        floor_alloc[cat] = min(math.floor(exact), len(bucket))
        remainders[cat] = exact - floor_alloc[cat]

    deficit = sample_size - sum(floor_alloc.values())
    sorted_by_remainder = sorted(
        category_buckets,
        key=lambda cat: (remainders[cat], len(category_buckets[cat]), cat),
        reverse=True,
    )

    final_alloc = dict(floor_alloc)
    for cat in sorted_by_remainder:
        if deficit <= 0:
            break
        if final_alloc[cat] < len(category_buckets[cat]):
            final_alloc[cat] += 1
            deficit -= 1

    selected: list = []
    for cat, bucket in category_buckets.items():
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[:final_alloc[cat]])

    rng.shuffle(selected)
    return selected


def load_dossiers(
    path,
    mode: str = "demo",
    sample_size: int = 30,
    seed: int = 42,
    exclude_ids: set = None,
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

    if exclude_ids:
        cases = [c for c in cases if str(c.get("case_id", "")) not in exclude_ids]

    if mode == "full":
        result = cases
    elif mode == "demo":
        rng = random.Random(seed)
        result = _stratified_sample(cases, sample_size, rng)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'demo' or 'full'.")

    # ── Filter to valid PHMRC categories ─────────────────────────────────────
    valid_set = set(PHMRC_CATEGORIES)
    filtered = []
    skipped = []
    for c in result:
        gt = c.get("ground_truth", "").strip()
        if gt in valid_set:
            filtered.append(c)
        else:
            skipped.append((c.get("case_id", "?"), gt))
    if skipped:
        print(f"[WARN] Skipping {len(skipped)} case(s) with ground_truth outside 21 valid PHMRC categories:")
        for cid, gt in skipped:
            print(f"  case_id={cid} | ground_truth='{gt}'")
    result = filtered

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
