import time
import json
import re
from agents.agents import agent1_node
from agents.data_loader import load_dossiers

# Load a real case
data_path = "data/patient_dossiers.json"
cases = load_dossiers(data_path, mode="demo", sample_size=1)
case = cases[0]

mock_state = {
    "case_id": case["case_id"],
    "ground_truth": case["ground_truth"],
    "has_narrative": case["has_narrative"],
    "full_dossier": case["full_dossier"],
    "agent1_output": {},
}

print(f"Testing Agent 1 with OpenThinker on Case ID: {case['case_id']}...")
start = time.time()
try:
    result = agent1_node(mock_state)
    elapsed = time.time() - start
    print(f"\nTime taken: {elapsed:.2f} seconds")
    print("\nParsed Agent 1 Result:")
    print(json.dumps(result["agent1_output"], indent=2))
except Exception as e:
    print(f"Error: {e}")
