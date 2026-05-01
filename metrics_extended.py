import os
import json
import math
import collections
import pandas as pd
from agents.utils import PHMRC_CATEGORIES

def _safe_float(val):
    if pd.isna(val):
        return 0.0
    return float(val)

def main():
    predictions_path = "results/predictions.csv"
    agent_outputs_path = "results/agent_outputs.jsonl"
    out_txt_path = "results/metrics_extended.txt"
    out_json_path = "results/metrics_extended.json"

    if not os.path.exists(predictions_path):
        print(f"Error: {predictions_path} not found.")
        return

    # 1. Read files
    df = pd.read_csv(predictions_path)
    total_cases = len(df)
    if total_cases == 0:
        print("No cases to evaluate.")
        return

    # Parse agent_outputs.jsonl
    agent_outputs = {}
    if os.path.exists(agent_outputs_path):
        with open(agent_outputs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    case_id = str(data.get("case_id", ""))
                    agent_outputs[case_id] = data
                except json.JSONDecodeError:
                    continue

    C = 21  # number of PHMRC categories
    
    # 1. Top-1 Accuracy
    # assume 'is_correct' or compute from ground_truth == mapped_category
    if "is_correct" in df.columns:
        accuracy_count = df["is_correct"].sum()
    else:
        accuracy_count = (df["ground_truth"] == df["mapped_category"]).sum()
    accuracy = accuracy_count / total_cases

    # 2. CCC
    ccc = (accuracy - (1.0 / C)) / (1.0 - (1.0 / C))

    # 3. PCCC (top-3 level)
    partial_correct = 0
    for idx, row in df.iterrows():
        case_id = str(row.get("case_id", ""))
        gt = row.get("ground_truth", "")
        mapped_cat = row.get("mapped_category", "")
        
        # from agent_outputs dict
        agent_data = agent_outputs.get(case_id, {})
        a2_data = agent_data.get("agent2", {})
        
        agent2_top3 = []
        if isinstance(a2_data, dict):
            agent2_top3 = a2_data.get("top3", [])
        
        # if not found in jsonl, try getting agent1/3 from dict or row
        agent1_diag = ""
        agent3_diag = ""
        
        if isinstance(agent_data.get("agent1"), dict):
            agent1_diag = agent_data["agent1"].get("diagnosis", "")
        if isinstance(agent_data.get("agent3"), dict):
            agent3_diag = agent_data["agent3"].get("diagnosis", "")
            
        # Merge logic
        candidates = [mapped_cat]
        if isinstance(agent2_top3, list):
            candidates.extend(agent2_top3)
        if agent1_diag:
            candidates.append(agent1_diag)
        if agent3_diag:
            candidates.append(agent3_diag)
            
        merged_top3 = []
        for c in candidates:
            if pd.notna(c) and str(c).strip() != "" and c not in merged_top3:
                merged_top3.append(c)
            if len(merged_top3) == 3:
                break
                
        if not merged_top3 and pd.notna(mapped_cat):
            merged_top3 = [mapped_cat]
            
        if gt in merged_top3:
            partial_correct += 1

    pccc = (partial_correct / total_cases - 1.0 / C) / (1.0 - 1.0 / C)

    # 4. CSMF Accuracy
    true_counts = collections.Counter(df["ground_truth"].dropna())
    pred_counts = collections.Counter(df["mapped_category"].dropna())
    
    true_fracs = {c: count / total_cases for c, count in true_counts.items()}
    pred_fracs = {c: count / total_cases for c, count in pred_counts.items()}
    
    csmf_error = 0.0
    for cat in PHMRC_CATEGORIES:
        true_f = true_fracs.get(cat, 0.0)
        pred_f = pred_fracs.get(cat, 0.0)
        csmf_error += abs(true_f - pred_f)
        
    min_true_frac = min(true_fracs.values()) if true_fracs else 0.0
    if (1.0 - min_true_frac) > 0:
        csmf_acc = 1.0 - csmf_error / (2.0 * (1.0 - min_true_frac))
    else:
        csmf_acc = 0.0

    # 5. Per-Category Metrics
    category_metrics = []
    
    for cat in PHMRC_CATEGORIES:
        if cat not in true_counts and cat not in pred_counts:
            continue
            
        gt_mask = df["ground_truth"] == cat
        pred_mask = df["mapped_category"] == cat
        
        tp = (gt_mask & pred_mask).sum()
        fp = (~gt_mask & pred_mask).sum()
        fn = (gt_mask & ~pred_mask).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        if support > 0 or fp > 0:
            category_metrics.append({
                "category": cat,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "correct": tp
            })

    category_metrics.sort(key=lambda x: x["support"], reverse=True)

    # 6. Macro & Weighted Averages
    supported_cats = [m for m in category_metrics if m["support"] > 0]
    macro_f1 = sum(m["f1"] for m in supported_cats) / len(supported_cats) if supported_cats else 0.0
    weighted_f1 = sum(m["f1"] * m["support"] for m in supported_cats) / total_cases if total_cases > 0 else 0.0

    # 7. Confusion Matrix and Top 5 Errors
    confusion_pairs = collections.Counter()
    for idx, row in df.iterrows():
        gt = row.get("ground_truth")
        pred = row.get("mapped_category")
        if pd.notna(gt) and pd.notna(pred) and gt != pred:
            confusion_pairs[(gt, pred)] += 1
            
    top5_errors = confusion_pairs.most_common(5)

    # 8. Subgroup Breakdowns
    subgroups = {}
    
    # a. Narrative vs No-Narrative
    if "has_narrative" in df.columns:
        df_narrative = df[df["has_narrative"] == 1]
        df_no_narrative = df[df["has_narrative"] == 0]
        
        narr_acc = (df_narrative["ground_truth"] == df_narrative["mapped_category"]).mean() if len(df_narrative) > 0 else 0.0
        no_narr_acc = (df_no_narrative["ground_truth"] == df_no_narrative["mapped_category"]).mean() if len(df_no_narrative) > 0 else 0.0
        
        subgroups["narrative"] = {"narrative_acc": narr_acc, "no_narrative_acc": no_narr_acc}

    # b. Per Broad Group
    if "broad_group" in df.columns:
        broad_groups_metrics = {}
        for bg in ["Infectious/Disease", "External/Trauma", "Chronic/Systemic/Other"]:
            df_bg = df[df["broad_group"] == bg]
            acc = (df_bg["ground_truth"] == df_bg["mapped_category"]).mean() if len(df_bg) > 0 else 0.0
            broad_groups_metrics[bg] = {"acc": acc, "count": len(df_bg)}
        subgroups["broad_groups"] = broad_groups_metrics

    # c. By Confidence Band
    if "confidence_score" in df.columns:
        df["conf_numeric"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0)
        df_high = df[df["conf_numeric"] >= 80]
        df_med = df[(df["conf_numeric"] >= 50) & (df["conf_numeric"] < 80)]
        df_low = df[df["conf_numeric"] < 50]
        
        high_acc = (df_high["ground_truth"] == df_high["mapped_category"]).mean() if len(df_high) > 0 else 0.0
        med_acc = (df_med["ground_truth"] == df_med["mapped_category"]).mean() if len(df_med) > 0 else 0.0
        low_acc = (df_low["ground_truth"] == df_low["mapped_category"]).mean() if len(df_low) > 0 else 0.0
        
        subgroups["confidence_bands"] = {
            "high": {"acc": high_acc, "count": len(df_high)},
            "medium": {"acc": med_acc, "count": len(df_med)},
            "low": {"acc": low_acc, "count": len(df_low)}
        }

    # d. Consensus vs Adjudicator
    if "final_reasoning" in df.columns:
        df_consensus = df[df["final_reasoning"].str.contains("Unanimous agent consensus", na=False)]
        df_adj = df[~df["final_reasoning"].str.contains("Unanimous agent consensus", na=False)]
        
        cons_acc = (df_consensus["ground_truth"] == df_consensus["mapped_category"]).mean() if len(df_consensus) > 0 else 0.0
        adj_acc = (df_adj["ground_truth"] == df_adj["mapped_category"]).mean() if len(df_adj) > 0 else 0.0
        cons_rate = len(df_consensus) / total_cases if total_cases > 0 else 0.0
        
        subgroups["consensus"] = {
            "consensus_acc": cons_acc,
            "adjudicator_acc": adj_acc,
            "consensus_rate": cons_rate
        }

    # 9. Per-Agent Accuracy
    agent_metrics = {}
    for agent_col_idx, agent_name in enumerate(["agent1", "agent2", "agent3"], 1):
        # Calculate accuracy assuming agent outputs are in the dict
        correct_count = 0
        agent_cats = []
        for idx, row in df.iterrows():
            case_id = str(row.get("case_id", ""))
            gt = row.get("ground_truth", "")
            ans = ""
            if case_id in agent_outputs and isinstance(agent_outputs[case_id].get(agent_name), dict):
                ans = agent_outputs[case_id][agent_name].get("diagnosis", "")
            if ans == gt and pd.notna(gt):
                correct_count += 1
            agent_cats.append(ans)
            
        acc = correct_count / total_cases if total_cases > 0 else 0.0
        
        # calculate macro f1
        cat_f1s = []
        for cat in PHMRC_CATEGORIES:
            tp = sum(1 for gt, ans in zip(df["ground_truth"], agent_cats) if gt == cat and ans == cat)
            fp = sum(1 for gt, ans in zip(df["ground_truth"], agent_cats) if gt != cat and ans == cat)
            fn = sum(1 for gt, ans in zip(df["ground_truth"], agent_cats) if gt == cat and ans != cat)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            cat_f1s.append(f1)
            
        m_f1 = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0.0
        agent_metrics[f"Agent {agent_col_idx}"] = {"accuracy": acc, "macro_f1": m_f1}

    # 10. Winning Agent Distribution
    winning_distributions = {}
    if "winning_agent" in df.columns:
        win_counts = collections.Counter(df["winning_agent"].fillna("Consensus/None"))
        winning_distributions = {str(k): v / total_cases for k, v in win_counts.items()}

    # Print / Write Output
    txt_lines = []
    txt_lines.append("==============================================================")
    txt_lines.append("  EXTENDED EVALUATION METRICS")
    txt_lines.append("==============================================================")
    txt_lines.append(f"Dataset : {predictions_path}")
    txt_lines.append(f"Cases   : {total_cases}  |  Categories : {C}  |  Seed : (as run)")
    txt_lines.append("")
    txt_lines.append("--- INDIVIDUAL-LEVEL METRICS ---")
    txt_lines.append(f"Top-1 Accuracy          : {accuracy*100:.2f}%  ({accuracy_count}/{total_cases})")
    txt_lines.append(f"CCC                     : {ccc:.3f}")
    txt_lines.append(f"PCCC (top-3)            : {pccc:.3f}")
    txt_lines.append("")
    txt_lines.append("--- POPULATION-LEVEL METRICS ---")
    txt_lines.append(f"CSMF Accuracy           : {csmf_acc:.3f}")
    txt_lines.append("")
    txt_lines.append("--- PER-AGENT ACCURACY ---")
    agent_roles = {
        "Agent 1": "Evidence Collector",
        "Agent 2": "Symptom Scorer",
        "Agent 3": "Timeline Analyst"
    }
    for agnt, metrics in agent_metrics.items():
        role = agent_roles.get(agnt, "Agent")
        txt_lines.append(f"{agnt} ({role}) : {metrics['accuracy']*100:.2f}%  |  Macro F1: {metrics['macro_f1']:.3f}")
    txt_lines.append("")
    txt_lines.append("--- PER-CATEGORY METRICS ---")
    txt_lines.append(f"{'Category':<38} | {'Precision':<9} | {'Recall':<6} | {'F1':<5} | {'Support':<7} | {'Correct'}")
    txt_lines.append("-" * 85)
    for m in category_metrics:
        txt_lines.append(f"{m['category']:<38} | {m['precision']:.3f}     | {m['recall']:.3f}  | {m['f1']:.3f} | {m['support']:<7} | {m['correct']}")
    txt_lines.append("")
    txt_lines.append("--- MACRO / WEIGHTED AVERAGES ---")
    txt_lines.append(f"Macro F1    : {macro_f1:.3f}")
    txt_lines.append(f"Weighted F1 : {weighted_f1:.3f}")
    txt_lines.append("")
    txt_lines.append("--- TOP-5 CONFUSION PAIRS ---")
    for pair, count in top5_errors:
        txt_lines.append(f"[{pair[0]} \u2192 {pair[1]} : {count} cases]")
    txt_lines.append("")
    txt_lines.append("--- SUBGROUP BREAKDOWN ---")
    if "narrative" in subgroups:
        txt_lines.append(f"Narrative Acc    : {subgroups['narrative']['narrative_acc']*100:.2f}%")
        txt_lines.append(f"No Narrative Acc : {subgroups['narrative']['no_narrative_acc']*100:.2f}%")
    if "broad_groups" in subgroups:
        for bg, stats in subgroups['broad_groups'].items():
            txt_lines.append(f"{bg:<25}: {stats['acc']*100:.2f}% ({stats['count']} cases)")
    if "confidence_bands" in subgroups:
        txt_lines.append("Confidence Bands:")
        for band, stats in subgroups['confidence_bands'].items():
            txt_lines.append(f"  {band.capitalize():<6} : {stats['acc']*100:.2f}% ({stats['count']} cases)")
    txt_lines.append("")
    txt_lines.append("--- CONSENSUS ANALYSIS ---")
    if "consensus" in subgroups:
        txt_lines.append(f"Consensus rate   : {subgroups['consensus']['consensus_rate']*100:.2f}%")
        txt_lines.append(f"Consensus acc    : {subgroups['consensus']['consensus_acc']*100:.2f}%")
        txt_lines.append(f"Adjudicator acc  : {subgroups['consensus']['adjudicator_acc']*100:.2f}%")
    
    txt_lines.append("\n--- WINNING AGENT DISTRIBUTION ---")
    for agent, percentage in winning_distributions.items():
        if not agent.strip():
            agent = "None/Consensus"
        txt_lines.append(f"{agent:<15} : {percentage*100:.2f}%")

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    json_output = {
        "accuracy": accuracy,
        "CCC": ccc,
        "PCCC": pccc,
        "CSMF_accuracy": csmf_acc,
        "per_category": category_metrics,
        "subgroups": subgroups,
        "confusion_top5": [{"ground_truth": k[0], "predicted": k[1], "count": v} for k, v in top5_errors],
        "agent_accuracy": agent_metrics,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "winning_agent_dist": winning_distributions
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            return super(NpEncoder, self).default(obj)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4, cls=NpEncoder)

    print(f"Metrics extended reports generated: {out_txt_path} and {out_json_path}")


if __name__ == "__main__":
    main()
