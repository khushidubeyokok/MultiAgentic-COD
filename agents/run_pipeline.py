"""
agents/run_pipeline.py
----------------------
Main entry point for the Verbal Autopsy multi-agent pipeline.
"""

import csv
import os
import sys
import time
from pathlib import Path

# ── Ensure the repo root (MultiAgentic-COD/) is on sys.path ──────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
MODE                = "demo"  # "demo" for stratified sample | "full" for all cases
SAMPLE_SIZE         = 10      # used only when MODE == "demo"
DELAY_BETWEEN_CASES = 1       # seconds between cases (local, no rate limits)
RANDOM_SEED         = 42      # fixed seed for reproducible results

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent
_DATA     = _ROOT / "data" / "patient_dossiers.json"
_RESULTS  = _ROOT / "results"
_PRED_CSV = _RESULTS / "predictions.csv"
_METRICS  = _RESULTS / "metrics_summary.txt"
_FAILED   = _RESULTS / "failed_cases.txt"

# ── CSV column order ──────────────────────────────────────────────────────────
_CSV_COLUMNS = [
    "case_id", "ground_truth", "has_narrative",
    "agent1_diagnosis", "agent1_confidence", "agent1_reasoning",
    "agent2_diagnosis", "agent2_confidence", "agent2_reasoning",
    "agent3_diagnosis", "agent3_confidence", "agent3_reasoning",
    "critic_critique",
    "final_diagnosis", "mapped_category", "confidence_score", "final_reasoning",
    "winning_agent",
    "is_correct", "agent1_correct", "agent2_correct", "agent3_correct",
]

from agents.utils import PHMRC_CATEGORIES

# ── Helper functions ──────────────────────────────────────────────────────────

def _agent_diag(output: dict) -> str:
    if not output or output.get("error"): return "Parse Error"
    return str(output.get("diagnosis", "Parse Error"))

def _agent_conf(output: dict) -> str:
    if not output or output.get("error"): return "N/A"
    return str(output.get("confidence", "N/A"))

def _agent_reason(output: dict) -> str:
    if not output or output.get("error"): return "N/A"
    return str(output.get("primary_reasoning", "N/A"))

def _is_correct(predicted: str, ground_truth: str) -> int:
    return 1 if str(predicted).strip().lower() == str(ground_truth).strip().lower() else 0

def _print_case_result(state: dict) -> None:
    case_id      = state.get("case_id", "?")
    ground_truth = state.get("ground_truth", "?")
    a1_diag = _agent_diag(state.get("agent1_output", {}))
    a1_conf = _agent_conf(state.get("agent1_output", {}))
    a2_diag = _agent_diag(state.get("agent2_output", {}))
    a2_conf = _agent_conf(state.get("agent2_output", {}))
    a3_diag = _agent_diag(state.get("agent3_output", {}))
    a3_conf = _agent_conf(state.get("agent3_output", {}))
    critique = state.get("critique", "")
    mapped_cat = state.get("mapped_category", "Unknown")
    conf_score = state.get("confidence_score", 0)
    match_str = "YES ✓" if _is_correct(mapped_cat, ground_truth) else "NO ✗"

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
    print(f"REASONING: {state.get('final_reasoning', 'N/A')}")
    print(f"MATCH: {match_str}  (Ground truth: {ground_truth})")
    print("=" * 60)
    print()

def _build_csv_row(state: dict) -> dict:
    ground_truth = state.get("ground_truth", "")
    a1 = state.get("agent1_output", {})
    a2 = state.get("agent2_output", {})
    a3 = state.get("agent3_output", {})
    mapped_cat = state.get("mapped_category", "")

    return {
        "case_id":           state.get("case_id", ""),
        "ground_truth":      ground_truth,
        "has_narrative":     int(bool(state.get("has_narrative", False))),
        "agent1_diagnosis":  _agent_diag(a1),
        "agent1_confidence": _agent_conf(a1),
        "agent1_reasoning":  _agent_reason(a1),
        "agent2_diagnosis":  _agent_diag(a2),
        "agent2_confidence": _agent_conf(a2),
        "agent2_reasoning":  _agent_reason(a2),
        "agent3_diagnosis":  _agent_diag(a3),
        "agent3_confidence": _agent_conf(a3),
        "agent3_reasoning":  _agent_reason(a3),
        "critic_critique":   state.get("critique", ""),
        "final_diagnosis":   state.get("final_diagnosis", ""),
        "mapped_category":   mapped_cat,
        "confidence_score":  state.get("confidence_score", 0),
        "final_reasoning":   state.get("final_reasoning", ""),
        "winning_agent":     state.get("winning_agent", "Adjudicator"),
        "is_correct":        _is_correct(mapped_cat, ground_truth),
        "agent1_correct":    _is_correct(_agent_diag(a1), ground_truth),
        "agent2_correct":    _is_correct(_agent_diag(a2), ground_truth),
        "agent3_correct":    _is_correct(_agent_diag(a3), ground_truth),
    }

def _append_to_csv(rows: list, path: Path, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def _log_failed(case_id: str, reason: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"case_id={case_id} | reason={reason}\n")

def _compute_and_print_metrics(rows: list, output_path: Path) -> None:
    if not rows: return
    total = len(rows)
    lines = []
    def _out(line: str = ""):
        print(line); lines.append(line)

    _out("=" * 60)
    _out("  EVALUATION METRICS")
    _out("=" * 60)
    correct_total = sum(int(r["is_correct"]) for r in rows)
    _out(f"\na. Overall Top-1 Accuracy: {correct_total}/{total} = {correct_total/total:.1%}")
    
    a1_c = sum(int(r["agent1_correct"]) for r in rows)
    a2_c = sum(int(r["agent2_correct"]) for r in rows)
    a3_c = sum(int(r["agent3_correct"]) for r in rows)
    _out(f"\nb. Per-Agent Accuracy:")
    _out(f"   Agent 1: {a1_c/total:.1%} | Agent 2: {a2_c/total:.1%} | Agent 3: {a3_c/total:.1%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

def main() -> None:
    from agents.data_loader import load_dossiers
    from agents.graph import run_single_case
    from agents.few_shot_examples import init_example_bank, get_excluded_case_ids

    # Build the few-shot example bank from the dataset
    init_example_bank(str(_DATA))
    excluded_ids = get_excluded_case_ids()

    cases = load_dossiers(str(_DATA), mode=MODE, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)
    # Remove example bank cases from evaluation (prevent data leakage)
    cases = [c for c in cases if str(c.get("case_id", "")) not in excluded_ids]
    total_cases = len(cases)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    for f in (_PRED_CSV, _FAILED):
        if f.exists(): f.unlink()

    csv_buffer = []
    first_write = True
    start_time = time.time()

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", f"idx_{idx}"))
        if not case.get("full_dossier", "").strip(): continue

        try:
            final_state = run_single_case(case)
            _print_case_result(final_state)
            row = _build_csv_row(final_state)
            csv_buffer.append(row)
            _append_to_csv(csv_buffer, _PRED_CSV, write_header=first_write)
            first_write = False
            csv_buffer = []
        except Exception as exc:
            _log_failed(case_id, str(exc), _FAILED)
            continue

        if idx % 5 == 0 or idx == total_cases:
            elapsed = time.time() - start_time
            avg = elapsed / idx
            eta = avg * (total_cases - idx)
            print(f"[PROGRESS] {idx}/{total_cases} | ETA: {eta/60:.1f} min")

        if idx < total_cases: time.sleep(DELAY_BETWEEN_CASES)

    all_rows = []
    if _PRED_CSV.exists():
        with open(_PRED_CSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader: all_rows.append(r)
    _compute_and_print_metrics(all_rows, _METRICS)

if __name__ == "__main__":
    main()
