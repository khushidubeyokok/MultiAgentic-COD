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
MODE                = "full"  # "demo" for stratified sample | "full" for all cases
SAMPLE_SIZE         = 300      # used only when MODE == "demo"
DELAY_BETWEEN_CASES = 1       # seconds between cases
RANDOM_SEED         = None      # fixed seed for reproducibility

# ── Pinned cases (set to a list of case_id strings to run ONLY those cases) ───
PINNED_CASE_IDS: list = []

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent
_DATA     = _ROOT / "data" / "patient_dossiers.json"
_RESULTS     = _ROOT / "results"
_PRED_CSV    = _RESULTS / "predictions.csv"
_METRICS     = _RESULTS / "metrics_summary.txt"
_FAILED      = _RESULTS / "failed_cases.txt"
_AGENT_LOG   = _RESULTS / "agent_outputs.jsonl"

# ── CSV column order ──────────────────────────────────────────────────────────
_CSV_COLUMNS = [
    "case_id", "broad_group", "ground_truth", "has_narrative",
    "agent1_diagnosis", "agent1_confidence", "agent1_reasoning",
    "agent2_diagnosis", "agent2_confidence", "agent2_reasoning",
    "agent3_diagnosis", "agent3_confidence", "agent3_reasoning",
    "final_diagnosis", "mapped_category", "confidence_score", "final_reasoning",
    "winning_agent",
    "is_correct", "agent1_correct", "agent2_correct", "agent3_correct",
]

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
    broad_group  = state.get("broad_group", "Unknown")
    ground_truth = state.get("ground_truth", "?")
    a1_diag = _agent_diag(state.get("agent1_output", {}))
    a1_conf = _agent_conf(state.get("agent1_output", {}))
    a2_diag = _agent_diag(state.get("agent2_output", {}))
    a2_conf = _agent_conf(state.get("agent2_output", {}))
    a3_diag = _agent_diag(state.get("agent3_output", {}))
    a3_conf = _agent_conf(state.get("agent3_output", {}))
    mapped_cat = state.get("mapped_category", "Unknown")
    conf_score = state.get("confidence_score", 0)
    match_str = "YES ✓" if _is_correct(mapped_cat, ground_truth) else "NO ✗"

    print("=" * 60)
    print(f"Case ID: {case_id}  |  Triage: {broad_group}  |  Ground Truth: {ground_truth}")
    print("-" * 60)
    print(f"Agent 1 (Evidence Collector): {a1_diag} [{a1_conf}]")
    print(f"Agent 2 (Symptom Scorer):     {a2_diag} [{a2_conf}]")
    print(f"Agent 3 (Timeline Analyst):   {a3_diag} [{a3_conf}]")
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
        "broad_group":       state.get("broad_group", ""),
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

def _write_agent_log(state: dict, path: Path) -> None:
    import json as _json
    a1 = state.get("agent1_output", {})
    a2 = state.get("agent2_output", {})
    a3 = state.get("agent3_output", {})

    record = {
        "case_id":       state.get("case_id", ""),
        "ground_truth":  state.get("ground_truth", ""),
        "agent1": {
            "diagnosis": a1.get("diagnosis", "Unknown"),
            "confidence":a1.get("confidence", "N/A"),
            "reasoning": a1.get("primary_reasoning", ""),
            "alternative_rejected": a1.get("alternative_rejected", ""),
            "rejection_reason":     a1.get("rejection_reason", ""),
        },
        "agent2": {
            "diagnosis": a2.get("diagnosis", "Unknown"),
            "confidence":a2.get("confidence", "N/A"),
            "reasoning": a2.get("primary_reasoning", ""),
            "top3":      a2.get("top3", []),
        },
        "agent3": {
            "diagnosis": a3.get("diagnosis", "Unknown"),
            "confidence":a3.get("confidence", "N/A"),
            "reasoning": a3.get("primary_reasoning", ""),
            "timeline_duration": a3.get("timeline_duration", ""),
        },
        "final": {
            "mapped_category": state.get("mapped_category", ""),
            "confidence_score": state.get("confidence_score", 0),
            "final_reasoning":  state.get("final_reasoning", ""),
            "winning_agent":    state.get("winning_agent", ""),
        },
        "is_correct": _is_correct(state.get("mapped_category",""), state.get("ground_truth","")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(record, ensure_ascii=False) + "\n")

def _compute_and_print_metrics(rows: list, output_path: Path) -> None:
    if not rows: return
    total = len(rows)
    lines = []
    def _out(line: str = ""):
        print(line); lines.append(line)

    _out("=" * 60)
    _out("  EVALUATION METRICS")
    _out("=" * 60)
    correct_total = sum(int(r.get("is_correct", 0)) for r in rows)
    _out(f"\na. Overall Top-1 Accuracy: {correct_total}/{total} = {correct_total/total:.1%}")
    
    a1_c = sum(int(r.get("agent1_correct", 0)) for r in rows)
    a2_c = sum(int(r.get("agent2_correct", 0)) for r in rows)
    a3_c = sum(int(r.get("agent3_correct", 0)) for r in rows)
    _out(f"\nb. Per-Agent Accuracy:")
    _out(f"   Agent 1: {a1_c/total:.1%} | Agent 2: {a2_c/total:.1%} | Agent 3: {a3_c/total:.1%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

def main() -> None:
    from agents.data_loader import load_dossiers
    from agents.graph import run_single_case

    # Load assessment cases
    cases = load_dossiers(str(_DATA), mode=MODE, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)

    # Filter to pinned cases if specified
    if PINNED_CASE_IDS:
        pinned_set = {str(c).strip() for c in PINNED_CASE_IDS}
        cases = [c for c in cases if str(c.get("case_id", "")).strip() in pinned_set]
        if not cases:
            all_cases = load_dossiers(str(_DATA), mode="full", sample_size=0, seed=RANDOM_SEED)
            cases = [c for c in all_cases if str(c.get("case_id", "")).strip() in pinned_set]

    total_cases = len(cases)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    for f in (_PRED_CSV, _FAILED, _AGENT_LOG):
        if f.exists(): f.unlink()

    csv_buffer = []
    first_write = True
    correct_count = 0
    start_time = time.time()

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", f"idx_{idx}"))
        if not case.get("full_dossier", "").strip(): continue

        try:
            final_state = run_single_case(case)
            _print_case_result(final_state)
            row = _build_csv_row(final_state)
            correct_count += int(row["is_correct"])
            csv_buffer.append(row)
            _append_to_csv(csv_buffer, _PRED_CSV, write_header=first_write)
            _write_agent_log(final_state, _AGENT_LOG)
            first_write = False
            csv_buffer = []
        except Exception as exc:
            _log_failed(case_id, str(exc), _FAILED)
            print(f"[ERROR] Case {case_id} failed: {exc}")
            continue

        if idx % 5 == 0 or idx == total_cases:
            elapsed = time.time() - start_time
            avg = elapsed / idx
            eta = avg * (total_cases - idx)
            acc = correct_count / idx
            print(f"[PROGRESS] {idx}/{total_cases} | Accuracy: {correct_count}/{idx} ({acc:.1%}) | ETA: {eta/60:.1f} min")

        if idx < total_cases: time.sleep(DELAY_BETWEEN_CASES)

    all_rows = []
    if _PRED_CSV.exists():
        with open(_PRED_CSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader: all_rows.append(r)
    _compute_and_print_metrics(all_rows, _METRICS)

if __name__ == "__main__":
    main()
