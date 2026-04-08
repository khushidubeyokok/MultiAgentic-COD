"""
agents/run_pipeline.py
----------------------
Main entry point for the Verbal Autopsy multi-agent pipeline.

Run from the MultiAgentic-COD project root:
    python agents/run_pipeline.py
  or equivalently:
    python -m agents.run_pipeline

Configuration constants at the top of the file control mode, sample size,
and inter-case delay to respect Groq rate limits.

Outputs (all relative to MultiAgentic-COD root):
    results/predictions.csv      — per-case results
    results/metrics_summary.txt  — evaluation metrics
    results/failed_cases.txt     — cases that failed completely
"""

import csv
import os
import sys
import time
from pathlib import Path

# ── Ensure the repo root (MultiAgentic-COD/) is on sys.path ──────────────────
# When invoked as `python agents/run_pipeline.py`, Python inserts agents/ into
# sys.path instead of the project root, breaking `from agents.X import Y`.
# We fix this by explicitly inserting the parent of this file's directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
MODE                = "demo"  # "demo" for stratified sample | "full" for all cases
SAMPLE_SIZE         = 10      # used only when MODE == "demo"
DELAY_BETWEEN_CASES = 1       # seconds between cases (Gemini free tier is very generous)
RANDOM_SEED         = None      # change this or set to None for a different sample each time

# ── Paths — all relative to MultiAgentic-COD (the repo root) ─────────────────
# agents/run_pipeline.py lives inside agents/, so .parent.parent == repo root
_ROOT     = Path(__file__).resolve().parent.parent
_DATA     = _ROOT / "data" / "patient_dossiers.json"
_RESULTS  = _ROOT / "results"
_PRED_CSV = _RESULTS / "predictions.csv"
_METRICS  = _RESULTS / "metrics_summary.txt"
_FAILED   = _RESULTS / "failed_cases.txt"

# ── CSV column order ──────────────────────────────────────────────────────────
_CSV_COLUMNS = [
    "case_id", "ground_truth", "has_narrative",
    "agent1_diagnosis", "agent1_confidence",
    "agent2_diagnosis", "agent2_confidence",
    "agent3_diagnosis", "agent3_confidence",
    "final_diagnosis", "mapped_category", "confidence_score",
    "winning_agent",
    "is_correct", "agent1_correct", "agent2_correct", "agent3_correct",
]

# ── 21 PHMRC categories (for post-run validation logging) ────────────────────
PHMRC_CATEGORIES = [
    "Drowning", "Poisonings", "Other Cardiovascular Diseases", "AIDS",
    "Violent Death", "Malaria", "Other Cancers", "Measles", "Meningitis",
    "Encephalitis", "Diarrhea/Dysentery", "Other Defined Causes of Child Deaths",
    "Other Infectious Diseases", "Hemorrhagic fever", "Other Digestive Diseases",
    "Bite of Venomous Animal", "Fires", "Falls", "Sepsis", "Pneumonia",
    "Road Traffic",
]


# ── Helper: safe diagnosis / confidence extraction ────────────────────────────

def _agent_diag(output: dict) -> str:
    """Return agent diagnosis, or 'Parse Error' if the output is an error dict."""
    if not output or output.get("error"):
        return "Parse Error"
    return str(output.get("diagnosis", "Parse Error"))


def _agent_conf(output: dict) -> str:
    """Return agent confidence, or 'N/A' for error dicts."""
    if not output or output.get("error"):
        return "N/A"
    return str(output.get("confidence", "N/A"))


def _is_correct(predicted: str, ground_truth: str) -> int:
    """Case-insensitive, whitespace-stripped comparison. Returns 1 or 0."""
    return 1 if predicted.strip().lower() == ground_truth.strip().lower() else 0


# ── Pretty-print a single case result to terminal ─────────────────────────────

def _print_case_result(state: dict) -> None:
    case_id      = state.get("case_id", "?")
    ground_truth = state.get("ground_truth", "?")
    a1_diag      = _agent_diag(state.get("agent1_output", {}))
    a1_conf      = _agent_conf(state.get("agent1_output", {}))
    a2_diag      = _agent_diag(state.get("agent2_output", {}))
    a2_conf      = _agent_conf(state.get("agent2_output", {}))
    a3_diag      = _agent_diag(state.get("agent3_output", {}))
    a3_conf      = _agent_conf(state.get("agent3_output", {}))
    critique     = state.get("critique", "")
    mapped_cat   = state.get("mapped_category", "?")
    conf_score   = state.get("confidence_score", 0)
    match        = _is_correct(mapped_cat, ground_truth)
    match_str    = "YES ✓" if match else "NO ✗"

    print("=" * 60)
    print(f"Case ID: {case_id}  |  Ground Truth: {ground_truth}")
    print("-" * 60)
    print(f"Agent 1 (Infectious Disease): {a1_diag} [{a1_conf}]")
    print(f"Agent 2 (Intensivist):        {a2_diag} [{a2_conf}]")
    print(f"Agent 3 (Trauma/Nutrition):   {a3_diag} [{a3_conf}]")
    print("-" * 60)
    critique_preview = (critique[:200] + "...") if len(critique) > 200 else critique
    print(f"CRITIQUE SUMMARY (first 200 chars): {critique_preview}")
    print("-" * 60)
    print(f"FINAL VERDICT: {mapped_cat} (confidence: {conf_score}/100)")
    print(f"MATCH: {match_str}  (Ground truth: {ground_truth})")
    print("=" * 60)
    print()


# ── Build a CSV row dict from a final state ───────────────────────────────────

def _build_csv_row(state: dict) -> dict:
    ground_truth = state.get("ground_truth", "")
    a1_diag      = _agent_diag(state.get("agent1_output", {}))
    a2_diag      = _agent_diag(state.get("agent2_output", {}))
    a3_diag      = _agent_diag(state.get("agent3_output", {}))
    mapped_cat   = state.get("mapped_category", "")

    return {
        "case_id":           state.get("case_id", ""),
        "ground_truth":      ground_truth,
        "has_narrative":     int(bool(state.get("has_narrative", False))),
        "agent1_diagnosis":  a1_diag,
        "agent1_confidence": _agent_conf(state.get("agent1_output", {})),
        "agent2_diagnosis":  a2_diag,
        "agent2_confidence": _agent_conf(state.get("agent2_output", {})),
        "agent3_diagnosis":  a3_diag,
        "agent3_confidence": _agent_conf(state.get("agent3_output", {})),
        "final_diagnosis":   state.get("final_diagnosis", ""),
        "mapped_category":   mapped_cat,
        "confidence_score":  state.get("confidence_score", 0),
        # winning_agent is returned by the adjudicator LLM but not a VAState field;
        # retrieve it from the raw state dict if LangGraph carried it through.
        "winning_agent":     state.get("winning_agent", ""),
        "is_correct":        _is_correct(mapped_cat, ground_truth),
        "agent1_correct":    _is_correct(a1_diag, ground_truth),
        "agent2_correct":    _is_correct(a2_diag, ground_truth),
        "agent3_correct":    _is_correct(a3_diag, ground_truth),
    }


# ── Incremental CSV append ────────────────────────────────────────────────────

def _append_to_csv(rows: list, path: Path, write_header: bool) -> None:
    """Append a list of row dicts to CSV. Creates file and parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _log_failed(case_id: str, reason: str, path: Path) -> None:
    """Append a failure record to the failed-cases log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"case_id={case_id} | reason={reason}\n")


# ── Evaluation metrics ────────────────────────────────────────────────────────

def _compute_and_print_metrics(rows: list, output_path: Path) -> None:
    """
    Compute and print all 7 evaluation metrics, then save to metrics_summary.txt.

    Metrics:
      a. Overall top-1 accuracy
      b. Per-agent accuracy (before adjudication)
      c. Adjudicator improvement over the best individual agent
      d. Per-category accuracy
      e. Narrative vs no-narrative accuracy
      f. Agent agreement rate (≥2 of 3 agree on same diagnosis)
      g. Adjudicator accuracy: agreement cases vs disagreement cases
    """
    if not rows:
        print("[WARN] No results to evaluate.")
        return

    total = len(rows)
    lines: list = []

    def _out(line: str = "") -> None:
        print(line)
        lines.append(line)

    _out("=" * 60)
    _out("  EVALUATION METRICS")
    _out("=" * 60)

    # ── a. Overall top-1 accuracy ─────────────────────────────────────────────
    correct_total = sum(r["is_correct"] for r in rows)
    overall_acc = correct_total / total
    _out(f"\na. Overall Top-1 Accuracy: {correct_total}/{total} = {overall_acc:.1%}")

    # ── b. Per-agent accuracy ─────────────────────────────────────────────────
    a1_correct = sum(r["agent1_correct"] for r in rows)
    a2_correct = sum(r["agent2_correct"] for r in rows)
    a3_correct = sum(r["agent3_correct"] for r in rows)
    a1_acc, a2_acc, a3_acc = a1_correct / total, a2_correct / total, a3_correct / total
    _out(f"\nb. Per-Agent Accuracy (before adjudication):")
    _out(f"   Agent 1 (Infectious Disease): {a1_correct}/{total} = {a1_acc:.1%}")
    _out(f"   Agent 2 (Intensivist):        {a2_correct}/{total} = {a2_acc:.1%}")
    _out(f"   Agent 3 (Trauma/Nutrition):   {a3_correct}/{total} = {a3_acc:.1%}")

    # ── c. Adjudicator improvement ────────────────────────────────────────────
    best_agent_acc = max(a1_acc, a2_acc, a3_acc)
    improvement = overall_acc - best_agent_acc
    _out(f"\nc. Adjudicator Improvement over Best Agent ({best_agent_acc:.1%}):")
    _out(f"   Adjudicator Accuracy: {overall_acc:.1%}  |  Delta: {improvement:+.1%}")

    # ── d. Per-category accuracy ──────────────────────────────────────────────
    _out(f"\nd. Per-Category Accuracy:")
    cat_totals: dict = {}
    cat_correct: dict = {}
    for r in rows:
        cat = r["ground_truth"] or "UNKNOWN"
        cat_totals[cat] = cat_totals.get(cat, 0) + 1
        cat_correct[cat] = cat_correct.get(cat, 0) + int(r["is_correct"])
    for cat in sorted(cat_totals):
        n, c = cat_totals[cat], cat_correct[cat]
        _out(f"   {cat:<42} {c}/{n} = {c/n:.1%}")

    # ── e. Narrative vs no-narrative accuracy ─────────────────────────────────
    narr_rows    = [r for r in rows if str(r["has_narrative"]) in ("1", "True", "true")]
    no_narr_rows = [r for r in rows if str(r["has_narrative"]) not in ("1", "True", "true")]
    _out(f"\ne. Narrative vs No-Narrative Accuracy:")
    if narr_rows:
        na = sum(int(r["is_correct"]) for r in narr_rows) / len(narr_rows)
        _out(f"   With narrative    (n={len(narr_rows):>3}): {na:.1%}")
    if no_narr_rows:
        nna = sum(int(r["is_correct"]) for r in no_narr_rows) / len(no_narr_rows)
        _out(f"   Without narrative (n={len(no_narr_rows):>3}): {nna:.1%}")

    # ── f. Agreement rate ─────────────────────────────────────────────────────
    agree_count = disagree_count = agree_correct = disagree_correct = 0
    for r in rows:
        diags = [r["agent1_diagnosis"], r["agent2_diagnosis"], r["agent3_diagnosis"]]
        valid  = [d for d in diags if d != "Parse Error"]
        agreed = any(valid.count(d) >= 2 for d in set(valid)) if len(valid) >= 2 else False
        if agreed:
            agree_count   += 1
            agree_correct += int(r["is_correct"])
        else:
            disagree_count   += 1
            disagree_correct += int(r["is_correct"])

    _out(f"\nf. Agent Agreement Rate: {agree_count}/{total} = {agree_count/total:.1%}")

    # ── g. Agreement vs disagreement accuracy ─────────────────────────────────
    _out(f"\ng. Adjudicator Accuracy: Agreed Cases vs Disagreed Cases:")
    if agree_count:
        _out(f"   Cases with ≥2 agents agreeing  (n={agree_count:>3}): {agree_correct/agree_count:.1%}")
    if disagree_count:
        _out(f"   Cases with no agent agreement  (n={disagree_count:>3}): {disagree_correct/disagree_count:.1%}")

    _out("\n" + "=" * 60)

    # ── Save metrics to file ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[INFO] Metrics saved to {output_path}")


# ── Main execution ────────────────────────────────────────────────────────────

def main() -> None:
    # Import here to avoid circular issues at module level
    from agents.data_loader import load_dossiers
    from agents.graph import run_single_case

    # ── Load dossiers ─────────────────────────────────────────────────────────
    cases = load_dossiers(str(_DATA), mode=MODE, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)
    total_cases = len(cases)

    # ── Prepare results directory; clear prior run files ─────────────────────
    _RESULTS.mkdir(parents=True, exist_ok=True)
    for f in (_PRED_CSV, _FAILED):
        if f.exists():
            f.unlink()

    csv_buffer: list = []
    first_write = True        # controls whether CSV header is written
    start_time = time.time()

    # ── Process each case ─────────────────────────────────────────────────────
    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", f"index_{idx}"))

        # Skip cases without a dossier to read
        if not case.get("full_dossier", "").strip():
            reason = "empty full_dossier"
            print(f"[SKIP] case_id={case_id}: {reason}")
            _log_failed(case_id, reason, _FAILED)
            continue

        # ── Run the full pipeline for this case ───────────────────────────────
        try:
            final_state = run_single_case(case)
        except Exception as exc:
            reason = f"pipeline exception: {exc}"
            print(f"[ERROR] case_id={case_id}: {reason}")
            _log_failed(case_id, reason, _FAILED)
            time.sleep(DELAY_BETWEEN_CASES)
            continue

        # ── Skip if ALL three agents errored ─────────────────────────────────
        all_errored = all(
            bool(final_state.get(f"agent{i}_output", {}).get("error"))
            for i in range(1, 4)
        )
        if all_errored:
            reason = "all three agents returned errors"
            print(f"[SKIP] case_id={case_id}: {reason}")
            _log_failed(case_id, reason, _FAILED)
            time.sleep(DELAY_BETWEEN_CASES)
            continue

        # ── Warn if mapped_category is not in PHMRC list ─────────────────────
        mapped_cat = final_state.get("mapped_category", "")
        if mapped_cat not in PHMRC_CATEGORIES:
            reason = f"mapped_category '{mapped_cat}' not in PHMRC list after fuzzy match"
            print(f"[WARN] case_id={case_id}: {reason}")
            _log_failed(case_id, reason, _FAILED)
            # Still record the result (is_correct will be 0)

        # ── Print to terminal ─────────────────────────────────────────────────
        _print_case_result(final_state)

        # ── Collect and save row immediately ───────────────────────────────
        row = _build_csv_row(final_state)
        csv_buffer.append(row)
        
        _append_to_csv(csv_buffer, _PRED_CSV, write_header=first_write)
        first_write = False
        csv_buffer = []   # flush buffer after each save to ensure visibility

        # Progress and ETA
        if idx % 5 == 0 or idx == total_cases:
            elapsed     = time.time() - start_time
            avg_per     = elapsed / idx
            eta_s       = avg_per * (total_cases - idx)
            print(f"[PROGRESS] Processed {idx}/{total_cases} cases | "
                  f"ETA: {eta_s/60:.1f} min")

        # ── Inter-case delay to respect rate limits ───────────────────────────
        if idx < total_cases:
            time.sleep(DELAY_BETWEEN_CASES)

    # ── Final flush of any remaining buffered rows ────────────────────────────
    if csv_buffer:
        _append_to_csv(csv_buffer, _PRED_CSV, write_header=first_write)

    print(f"\n[INFO] Results saved to {_PRED_CSV}")

    # ── Read all rows back for metric computation ─────────────────────────────
    all_rows: list = []
    if _PRED_CSV.exists():
        with open(_PRED_CSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Convert numeric fields from strings back to int
                row["is_correct"]     = int(row.get("is_correct", 0))
                row["agent1_correct"] = int(row.get("agent1_correct", 0))
                row["agent2_correct"] = int(row.get("agent2_correct", 0))
                row["agent3_correct"] = int(row.get("agent3_correct", 0))
                all_rows.append(row)

    # ── Compute and print evaluation metrics ──────────────────────────────────
    _compute_and_print_metrics(all_rows, _METRICS)


if __name__ == "__main__":
    main()
