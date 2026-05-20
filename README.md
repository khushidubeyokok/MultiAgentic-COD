# MultiAgentic-COD

A multi-agent pipeline for automated cause-of-death assignment using PHMRC Verbal Autopsy data (Child Module). This project leverages multiple specialist LLM agents to analyze structured survey data and caregiver narratives, producing both structured and flat clinical dossiers and assigning likely causes of death.

## Features
- **Translation Engine**: Converts raw survey and narrative data into clinical prose dossiers.
- **Preprocessing**: Cleans and annotates dossiers, ranks evidence domains, and highlights primary/secondary symptoms.
- **Multi-Agent Reasoning**: Three specialist agents (Evidence Collector, Symptom Scorer, Timeline Analyst) independently analyze each case.
- **Adjudication**: A final adjudicator agent reviews all outputs and makes the final cause-of-death assignment.
- **Metrics & Results**: Outputs predictions, agent logs, and accuracy metrics for evaluation.

## Directory Structure
- `agents/` — Core agent logic, graph pipeline, preprocessing, disease references, and utilities
- `data/` — Input data (e.g., `patient_dossiers.json`)
- `results/` — Output predictions, logs, and metrics
- `translation_engine.py` — Data translation and dossier generation
- `requirements.txt` — Python dependencies

## Usage
1. **Prepare Data**: Place raw survey and narrative CSVs in the project directory.
2. **Generate Dossiers**:
   ```sh
   python translation_engine.py
   ```
   This creates `data/patient_dossiers.json` for agent processing.
3. **Run Pipeline**:
   ```sh
   python agents/run_pipeline.py
   ```
   Results are saved in the `results/` folder.

## Requirements
- Python 3.10+
- See `requirements.txt` for dependencies (e.g., pandas, numpy, langchain, etc.)

## Customization
- Modify `agents/model_config.py` to change LLM model, sample size, or other parameters.
- Add or edit disease category references in `agents/disease_ref.py`.
