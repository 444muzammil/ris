# 🚀 RankMind — Recruiter Intelligence System

> **An advanced hybrid AI-powered candidate discovery pipeline built for the Redrob Hackathon Challenge.**

RankMind intelligently ranks candidate profiles against a job description using a **three-phase AI pipeline** that combines **hybrid retrieval, constraint-based reasoning, and explainable AI** to produce recruiter-friendly candidate rankings.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Data](#-data)
- [Output](#-output)
- [Results](#-results)
- [Reproducibility](#-reproducibility)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

# 📖 Overview

Recruiters often need to identify the best candidates from hundreds or thousands of resumes.

**RankMind** addresses this challenge using a multi-stage ranking pipeline that combines:

- 🔍 Hybrid semantic + lexical retrieval
- 📋 Constraint-based candidate filtering
- 🧠 Explainable AI scoring

Unlike simple keyword matching systems, RankMind balances **semantic similarity**, **hard hiring requirements**, and **human-readable explanations**, making the ranking process transparent and recruiter-friendly.

---

# ✨ Features

### 🔍 Phase 1 — Hybrid Retrieval

Combines:

- TF-IDF (scikit-learn)
- Sentence-BERT (`all-MiniLM-L6-v2`)

to capture both:

- Exact keyword matches
- Deep semantic similarity

---

### 📋 Phase 2 — Constraint Reasoning

Applies hiring logic including:

- ✅ Minimum years of experience
- ✅ Job title relevance
- ✅ Required technologies
- ✅ Soft preference scoring

---

### 🧠 Phase 3 — Explainable AI

Every candidate receives an explanation describing:

- Why they ranked highly
- Skills that matched
- Experience bonuses
- Missing requirements (if any)

This improves recruiter trust and decision making.

---

### 💻 Offline First

- No external APIs
- CPU-only execution
- Fully local inference

---

### 🔄 Reproducible

All weights, thresholds, and scoring logic are centralized inside:

```
src/ranking_engine.py
```

---

### 🏆 Hackathon Ready

Generates a submission file compatible with the Redrob Hackathon format:

```
submission.csv
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/444muzammil/ris.git
cd ris
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

### Required packages

- scikit-learn
- sentence-transformers
- pandas
- numpy
- pyyaml
- python-docx

---

# 🚀 Usage

Run the ranking pipeline:

```bash
python rank.py \
    --candidates ./data/candidates.jsonl \
    --out ./submission.csv
```

---

## Available Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--candidates` | Candidate JSONL file | `./data/candidates.jsonl` |
| `--out` | Output CSV | `./submission.csv` |
| `--job_desc` | Job Description DOCX | `./job_description.docx` |
| `--config` | YAML configuration | `./submission_metadata.yaml` |

> **Note:** If omitted, the default paths are automatically used.

---

# 🏗 Pipeline Architecture

## Phase 1 — Hybrid Retrieval

The first stage computes relevance using two independent retrieval methods.

### TF-IDF

Captures:

- Exact keywords
- Skill overlap
- Technology names

### Sentence-BERT

Encodes resumes and job descriptions into dense embeddings using:

```
all-MiniLM-L6-v2
```

Similarity is computed using cosine distance.

### Hybrid Score

The final retrieval score is:

```
Hybrid Score =
α × TF-IDF Score
+
β × Dense Embedding Score
```

---

## Phase 2 — Constraint-Based Reasoning

Candidates are evaluated using hiring constraints.

### Hard Constraints

Examples include:

- Minimum years of experience
- Required technologies
- Mandatory role eligibility

Candidates failing mandatory requirements receive penalties or are filtered.

### Soft Constraints

Additional score adjustments are applied for:

- Matching job titles
- Relevant experience
- Preferred technologies
- Domain expertise

---

## Phase 3 — Explainability Engine

Each ranked candidate includes a natural language explanation.

Example:

> Strong semantic similarity with the job description. Meets required experience. Demonstrates expertise in Python, Machine Learning, and NLP. Previous Software Engineer role closely matches target position.

This enables recruiters to understand **why** each score was assigned.

---

# 📁 Project Structure

```
.
├── data/
│   └── candidates.jsonl
│
├── src/
│   └── ranking_engine.py
│
├── job_description.docx
├── submission_metadata.yaml
├── rank.py
├── requirements.txt
└── README.md
```

---

# 📂 Data

## Candidate Profiles

```
data/candidates.jsonl
```

Each line contains a candidate record with fields such as:

- Candidate ID
- Name
- Experience
- Skills
- Job Titles
- Education

---

## Job Description

```
job_description.docx
```

The parser extracts:

- Required experience
- Technologies
- Preferred skills
- Job titles
- Keywords

---

## Configuration

```
submission_metadata.yaml
```

Defines:

- Scoring weights
- Constraint thresholds
- Explanation templates

---

# 📄 Output

The generated file:

```
submission.csv
```

contains:

| Column | Description |
|----------|-------------|
| `candidate_id` | Unique candidate identifier |
| `score` | Final ranking score |
| `rank` | Candidate position |
| `explanation` | Explainable AI justification |

---

# 📊 Results

The ranking pipeline prioritizes candidates who:

- Possess strong semantic similarity
- Meet mandatory hiring constraints
- Match desired technologies
- Have relevant job titles
- Receive transparent explanations

This produces recruiter-friendly rankings that balance **accuracy**, **fairness**, and **interpretability**.

---

# 🔬 Reproducibility

- ✅ CPU-only execution
- ✅ No internet dependency
- ✅ Fixed random seeds
- ✅ Fully deterministic pipeline
- ✅ Exact dependency versions listed in `requirements.txt`

---

# 🙏 Acknowledgments

Special thanks to:

- **Redrob Hackathon** organizers
- **scikit-learn**
- **Sentence Transformers**
- **PyYAML**
- The contributors of the candidate dataset and job description

---

# 📬 Contact

For questions, suggestions, or contributions:

- **GitHub:** https://github.com/444muzammil
- Open an issue in this repository.

---

## ⭐ If you found this project useful, consider giving it a star!

Made with ❤️ for the Redrob Hackathon Challenge.
