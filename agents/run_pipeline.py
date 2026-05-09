"""
agents/run_pipeline.py
----------------------
Main entry point for the Verbal Autopsy multi-agent pipeline.
"""

import collections
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ensure the repo root (MultiAgentic-COD/) is on sys.path ──────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
from agents.model_config import ACTIVE_PROFILE

MODE                = ACTIVE_PROFILE["sample_mode"]
SAMPLE_SIZE         = ACTIVE_PROFILE["sample_size"]
DELAY_BETWEEN_CASES = 0       # seconds between cases
RANDOM_SEED         = ACTIVE_PROFILE.get("seed", 42)  # fixed seed for reproducibility

# ── Pinned cases (set to a list of case_id strings to run ONLY those cases) ───
PINNED_CASE_IDS: list = []

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent
_DATA     = _ROOT / "data" / "patient_dossiers.json"
_RESULTS     = _ROOT / "results"
_PRED_CSV    = _RESULTS / "predictions.csv"
_METRICS     = _RESULTS / "metrics_final.txt"
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
            "parse_failure": bool(a1.get("parse_failure", False)),
            "raw_response": a1.get("raw_response", ""),
        },
        "agent2": {
            "diagnosis": a2.get("diagnosis", "Unknown"),
            "confidence":a2.get("confidence", "N/A"),
            "reasoning": a2.get("primary_reasoning", ""),
            "top3":      a2.get("top3", []),
            "parse_failure": bool(a2.get("parse_failure", False)),
            "raw_response": a2.get("raw_response", ""),
        },
        "agent3": {
            "diagnosis": a3.get("diagnosis", "Unknown"),
            "confidence":a3.get("confidence", "N/A"),
            "reasoning": a3.get("primary_reasoning", ""),
            "timeline_duration": a3.get("timeline_duration", ""),
            "parse_failure": bool(a3.get("parse_failure", False)),
            "raw_response": a3.get("raw_response", ""),
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

def _compute_and_save_all_metrics(rows: list, agent_outputs_path: Path, output_path: Path) -> None:
    if not rows:
        return

    from agents.utils import PHMRC_CATEGORIES, fuzzy_match_category

    total = len(rows)
    category_count = len(PHMRC_CATEGORIES)
    agent_outputs = {}
    if agent_outputs_path.exists():
        with open(agent_outputs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                agent_outputs[str(record.get("case_id", ""))] = record

    def _norm(value: object) -> str:
        resolved = fuzzy_match_category(str(value))
        return resolved or str(value or "").strip()

    def _correct(row: dict) -> bool:
        return _norm(row.get("ground_truth")) == _norm(row.get("mapped_category"))

    def _pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    correct_total = sum(1 for row in rows if _correct(row))
    accuracy = correct_total / total
    ccc = (accuracy - (1.0 / category_count)) / (1.0 - (1.0 / category_count))

    partial_correct = 0
    for row in rows:
        case_id = str(row.get("case_id", ""))
        gt = _norm(row.get("ground_truth"))
        agent_data = agent_outputs.get(case_id, {})
        agent2_top3 = agent_data.get("agent2", {}).get("top3", [])
        if not isinstance(agent2_top3, list):
            agent2_top3 = []

        candidates = [row.get("mapped_category")]
        candidates.extend(agent2_top3)
        candidates.append(agent_data.get("agent1", {}).get("diagnosis", row.get("agent1_diagnosis")))
        candidates.append(agent_data.get("agent3", {}).get("diagnosis", row.get("agent3_diagnosis")))

        merged_top3 = []
        for candidate in candidates:
            normalized = _norm(candidate)
            if normalized and normalized not in merged_top3:
                merged_top3.append(normalized)
            if len(merged_top3) == 3:
                break
        if gt in merged_top3:
            partial_correct += 1

    pccc_raw = partial_correct / total
    pccc = (pccc_raw - (1.0 / category_count)) / (1.0 - (1.0 / category_count))

    true_counts = collections.Counter(_norm(row.get("ground_truth")) for row in rows)
    pred_counts = collections.Counter(_norm(row.get("mapped_category")) for row in rows)
    csmf_error = 0.0
    for cat in PHMRC_CATEGORIES:
        csmf_error += abs(true_counts.get(cat, 0) / total - pred_counts.get(cat, 0) / total)
    min_true_frac = min((true_counts.get(cat, 0) / total for cat in PHMRC_CATEGORIES if true_counts.get(cat, 0)), default=0.0)
    csmf_acc = 1.0 - csmf_error / (2.0 * (1.0 - min_true_frac)) if min_true_frac < 1.0 else 0.0

    category_metrics = []
    for cat in PHMRC_CATEGORIES:
        tp = sum(1 for row in rows if _norm(row.get("ground_truth")) == cat and _norm(row.get("mapped_category")) == cat)
        fp = sum(1 for row in rows if _norm(row.get("ground_truth")) != cat and _norm(row.get("mapped_category")) == cat)
        fn = sum(1 for row in rows if _norm(row.get("ground_truth")) == cat and _norm(row.get("mapped_category")) != cat)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support or fp:
            category_metrics.append({
                "category": cat,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "correct": tp,
            })
    category_metrics.sort(key=lambda item: item["support"], reverse=True)
    supported = [item for item in category_metrics if item["support"]]
    macro_f1 = sum(item["f1"] for item in supported) / len(supported) if supported else 0.0
    weighted_f1 = sum(item["f1"] * item["support"] for item in supported) / total

    def _agent_metrics(agent_num: int) -> dict:
        pred_key = f"agent{agent_num}_diagnosis"
        correct = sum(1 for row in rows if _norm(row.get(pred_key)) == _norm(row.get("ground_truth")))
        f1s = []
        for cat in PHMRC_CATEGORIES:
            tp = sum(1 for row in rows if _norm(row.get("ground_truth")) == cat and _norm(row.get(pred_key)) == cat)
            fp = sum(1 for row in rows if _norm(row.get("ground_truth")) != cat and _norm(row.get(pred_key)) == cat)
            fn = sum(1 for row in rows if _norm(row.get("ground_truth")) == cat and _norm(row.get(pred_key)) != cat)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        return {"correct": correct, "accuracy": correct / total, "macro_f1": sum(f1s) / len(f1s)}

    agent_metrics = {idx: _agent_metrics(idx) for idx in (1, 2, 3)}

    confusion_pairs = collections.Counter()
    for row in rows:
        gt = _norm(row.get("ground_truth"))
        pred = _norm(row.get("mapped_category"))
        if gt and pred and gt != pred:
            confusion_pairs[(gt, pred)] += 1

    def _subset_accuracy(subset: list) -> float:
        return sum(1 for row in subset if _correct(row)) / len(subset) if subset else 0.0

    narrative_rows = [row for row in rows if str(row.get("has_narrative", "0")) in {"1", "True", "true"}]
    no_narrative_rows = [row for row in rows if row not in narrative_rows]
    broad_groups = sorted({str(row.get("broad_group", "")).strip() for row in rows if str(row.get("broad_group", "")).strip()})
    confidence_rows = []
    for row in rows:
        try:
            confidence = float(row.get("confidence_score", 0) or 0)
        except ValueError:
            confidence = 0.0
        confidence_rows.append((confidence, row))

    consensus_rows = [row for row in rows if "Unanimous agent consensus" in str(row.get("final_reasoning", ""))]
    adjudicator_rows = [row for row in rows if row not in consensus_rows]

    lines = []
    def _out(line: str = "") -> None:
        print(line)
        lines.append(line)

    _out("=" * 66)
    _out("  FINAL EVALUATION METRICS")
    _out("=" * 66)
    _out("")
    _out("Section 1 - Run Config")
    _out(f"Model name  : {ACTIVE_PROFILE['model']}")
    _out(f"Sample size : {SAMPLE_SIZE}")
    _out(f"Mode        : {MODE}")
    _out(f"Seed        : {RANDOM_SEED}")
    _out(f"Timestamp   : {datetime.now().isoformat(timespec='seconds')}")
    _out("")
    _out("Section 2 - Individual-Level Metrics")
    _out(f"Top-1 Accuracy : {correct_total}/{total} ({_pct(accuracy)})")
    _out(f"CCC            : {ccc:.3f}")
    _out(f"PCCC (top-3)   : {pccc:.3f} ({partial_correct}/{total})")
    _out("")
    _out("Section 3 - Population-Level Metrics")
    _out(f"CSMF Accuracy  : {csmf_acc:.3f}")
    _out("")
    _out("Section 4 - Per-Agent Accuracy")
    for idx, role in [(1, "Evidence Collector"), (2, "Symptom Scorer"), (3, "Timeline Analyst")]:
        metrics = agent_metrics[idx]
        _out(f"Agent {idx} ({role}): {metrics['correct']}/{total} ({_pct(metrics['accuracy'])}) | Macro F1: {metrics['macro_f1']:.3f}")
    _out("")
    _out("Section 5 - Per-Category Table")
    _out(f"{'category':<38} | {'precision':>9} | {'recall':>6} | {'f1':>5} | {'support':>7} | {'correct':>7}")
    _out("-" * 86)
    for item in category_metrics:
        _out(f"{item['category']:<38} | {item['precision']:>9.3f} | {item['recall']:>6.3f} | {item['f1']:>5.3f} | {item['support']:>7} | {item['correct']:>7}")
    _out("")
    _out("Section 6 - Macro / Weighted Averages")
    _out(f"Macro F1    : {macro_f1:.3f}")
    _out(f"Weighted F1 : {weighted_f1:.3f}")
    _out("")
    _out("Section 7 - Top-5 Confusion Pairs")
    for (gt, pred), count in confusion_pairs.most_common(5):
        _out(f"{gt} -> {pred}: {count}")
    if not confusion_pairs:
        _out("None")
    _out("")
    _out("Section 8 - Subgroup Breakdowns")
    _out(f"Narrative       : {_pct(_subset_accuracy(narrative_rows))} ({len(narrative_rows)} cases)")
    _out(f"No narrative    : {_pct(_subset_accuracy(no_narrative_rows))} ({len(no_narrative_rows)} cases)")
    for group in broad_groups:
        subset = [row for row in rows if str(row.get("broad_group", "")).strip() == group]
        _out(f"{group:<25}: {_pct(_subset_accuracy(subset))} ({len(subset)} cases)")
    bands = {
        "high (>=80)": [row for confidence, row in confidence_rows if confidence >= 80],
        "medium (50-79)": [row for confidence, row in confidence_rows if 50 <= confidence < 80],
        "low (<50)": [row for confidence, row in confidence_rows if confidence < 50],
    }
    for label, subset in bands.items():
        _out(f"Confidence {label:<14}: {_pct(_subset_accuracy(subset))} ({len(subset)} cases)")
    _out(f"Consensus       : {_pct(_subset_accuracy(consensus_rows))} ({len(consensus_rows)} cases)")
    _out(f"Adjudicator     : {_pct(_subset_accuracy(adjudicator_rows))} ({len(adjudicator_rows)} cases)")
    _out("")
    _out("Section 9 - Consensus Analysis")
    _out(f"Consensus rate  : {_pct(len(consensus_rows) / total)}")
    _out(f"Consensus acc   : {_pct(_subset_accuracy(consensus_rows))}")
    _out(f"Adjudicator acc : {_pct(_subset_accuracy(adjudicator_rows))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

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
    _compute_and_save_all_metrics(all_rows, _AGENT_LOG, _METRICS)

if __name__ == "__main__":
    main()
