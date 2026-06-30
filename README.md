RankMind — Recruiter Intelligence System
Download Python | Python.org The MIT License – Open Source Initiative

An advanced hybrid AI-powered candidate discovery pipeline built for the Redrob Hackathon Challenge. RankMind intelligently ranks candidate profiles against a job description using a three-phase pipeline that combines lexical/dense retrieval, constraint-based reasoning, and explainable AI.

[Image blocked: Pipeline Overview]

Table of Contents
Overview
Features
Installation
Usage
Pipeline Architecture
Data
Results
Reproducibility & Environment
License
Acknowledgments
Contact
Overview
RankMind addresses the challenge of efficiently identifying top talent from large candidate pools. By integrating state-of-the-art NLP techniques with rule-based reasoning, the system delivers transparent, explainable rankings that align with both semantic similarity and hard constraints (experience, title, tech stack).

Features
Hybrid Retrieval (Phase 1): Combines TF-IDF (scikit-learn) and dense embeddings (Sentence-BERT all-MiniLM-L6-v2) for robust semantic and lexical matching.
Constraint Reasoning (Phase 2): Enforces hard filters and soft preferences on experience years, job titles, and core technologies.
Explainable AI (Phase 3): Generates human-readable justifications for each candidate’s score, enhancing recruiter trust.
Offline-First: No external API calls; all computation runs locally on CPU.
Reproducible: All scoring weights and logic are encapsulated in src/ranking_engine.py.
Ready for Submission: Outputs a ranked CSV (submission.csv) that matches the expected hackathon format.
Installation
Clone the repository

Code
· bash
git clone https://github.com/444muzammil/ris.git
cd ris
Set up a Python environment (>=3.11)

Code
· bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

Code
· bash
pip install -r requirements.txt
The requirements.txt includes:

scikit-learn
sentence-transformers
pandas
numpy
pyyaml
python-docx (for reading the Job Description)
Usage
Run the ranking pipeline from the command line:

Code
· bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
Arguments
Argument	Description	Default
--candidates	Path to the candidate profiles (JSONL)	./data/candidates.jsonl
--out	Output CSV file path	./submission.csv
--job_desc	Path to the Job Description (DOCX)	./job_description.docx
--config	Path to scoring configuration (YAML)	./submission_metadata.yaml
Note: The script will automatically load the default files if paths are not provided.

Output
The generated submission.csv contains:

candidate_id: Unique identifier from the input
score: Final ranking score (higher = better)
rank: Position in the sorted list (1 = top)
explanation: Human-readable justification for the score
Pipeline Architecture
RankMind operates in three distinct phases:

Phase 1: Lexical/Dense Hybrid Retrieval
TF-IDF Vectorizer (scikit-learn) captures exact keyword matches.
Sentence-BERT (all-MiniLM-L6-v2) encodes semantic similarity.
Hybrid score = weighted sum of normalized TF-IDF and cosine similarity.
Phase 2: Constraint-Based Reasoning
Hard Filters: Eliminate candidates failing mandatory criteria (e.g., minimum years of experience).
Soft Scoring: Adjust scores based on title relevance and tech-stack match using fuzzy matching and predefined weights.
Phase 3: Explanation Engine
For each candidate, generates a dynamic explanation detailing:
Which components contributed to the score.
Any penalties or bonuses applied.
How the candidate matches the job description.
Data
Candidate Profiles: data/candidates.jsonl Each line is a JSON object representing a candidate with fields like id, name, experience_years, titles, skills, etc.
Job Description: job_description.docx Contains the target role description, which is parsed to extract keywords, required experience, desired titles, and tech stack.
Configuration: submission_metadata.yaml Defines weights for each phase, constraint thresholds, and explanation templates.
Results
The pipeline outputs a ranked list saved as submission.csv. Top candidates exhibit high semantic alignment, satisfy hard constraints, and receive transparent justifications.

Reproducibility & Environment
Compute: CPU-only; no GPU required.
Determinism: All random seeds are fixed in src/ranking_engine.py (if any stochastic components are used).
Version Control: Exact dependency versions are specified in requirements.txt.
Offline Operation: No network calls during execution; all models are loaded locally.
License
This project is licensed under the MIT License - see the LICENSE [blocked] file for details.

Acknowledgments
Thanks to the Redrob Hackathon organizers for the challenge.
The open-source community for scikit-learn, sentence-transformers, and PyYAML.
Thanks to the contributors of the candidate dataset and job description.
Contact
For questions or feedback, please open an issue in this repository or contact the maintainers at 444muzammil (Muzammil) · GitHub.

Made with ❤️ for the Redrob Hackathon Challenge.
