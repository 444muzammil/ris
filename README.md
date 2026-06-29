Markdown
# RankMind — Recruiter Intelligence System

An advanced hybrid AI-powered candidate discovery pipeline built for the Redrob Hackathon Challenge.

🚀 How to Run
This project uses a custom dual-layer ranking pipeline (rank.py) to process candidate data against our defined Job Description.

Prerequisites
Ensure Python 3.11+ is installed.

Install the required dependencies:

Bash
pip install -r requirements.txt
Execution
Run the ranking engine from your terminal using the following command. The script will automatically load the candidate data and output a scored CSV file:

Bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
Pipeline Overview
Phase 1 (Lexical/Dense Hybrid): Utilizes scikit-learn (TF-IDF) and sentence-transformers (all-MiniLM-L6-v2) to compare candidate profiles against the Job Description.

Phase 2 (Reasoning Engine): Processes structural constraints (experience years, title requirements, and core tech stacks).

Phase 3 (Explanation Engine): Generates factual, dynamic justifications for each ranking score to provide transparency to recruiters.

Environment & Reproducibility
Compute: This project is optimized for CPU-based inference. No GPU is required for standard execution.

Constraints: The system is designed for offline operation; no API calls or network connectivity are required during the ranking process.

Reproducibility: All scoring weights and behavioral modifiers are contained within src/ranking_engine.py, ensuring consistent scoring across different environments.
