import argparse
import pandas as pd
import os

from src.ranking_engine import RankingEngine
from src.explanation_engine import ExplanationEngine

def get_jd_text():
    try:
        from docx import Document
        if os.path.exists("job_description.docx"):
            doc = Document("job_description.docx")
            return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        pass

    if os.path.exists("job_description.txt"):
        with open("job_description.txt", "r", encoding="utf-8") as f:
            return f.read()

    return "Founding AI Engineer with 5 to 9 years of experience. Must have strong skills in Python, Machine Learning, RAG, NLP, and Sentence Transformers."

def main():
    parser = argparse.ArgumentParser(description="RankMind Pipeline CLI")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to the candidates JSONL file")
    parser.add_argument("--out", type=str, default="./submission.csv", help="Path to save the final CSV")
    args = parser.parse_args()

    print("🚀 Starting RankMind Pipeline...")

    print(f"📦 Loading dataset from: {args.candidates}")
    try:
        df = pd.read_json(args.candidates, lines=True)
    except Exception as e:
        print(f"❌ Error loading candidates: {e}")
        return

    jd_text = get_jd_text()

    print("🧠 Running Phase 1 & 2: Hybrid AI Ranking Engine...")
    ranker = RankingEngine(job_description=jd_text)
    top_100_df = ranker.process_and_rank(df)

    print("✍️ Running Phase 3: Factual Explanation Engine...")
    explainer = ExplanationEngine()
    final_df = explainer.generate_explanations(top_100_df)

    print(f"💾 Saving final output to: {args.out}")
    final_df.to_csv(args.out, index=False)
    print("✅ Pipeline execution complete!")

if __name__ == "__main__":
    main()
