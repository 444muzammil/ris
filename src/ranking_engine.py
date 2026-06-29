import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- GLOBAL MODEL CACHE ---
_AI_MODEL_CACHE = None

def get_cached_model():
  """Ensures the heavy AI model is loaded into memory exactly ONCE and from a local, offline path."""
    global _AI_MODEL_CACHE
    if _AI_MODEL_CACHE is None:
        # 1. Get the directory where this script (ranking_engine.py) is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Build the path to the model folder. 
        # If 'local_minilm_model' is in the root (next to 'rank.py'), 
        # use '..' to go up one level from 'src'
        model_path = os.path.join(script_dir, 'local_minilm_model')
        
        # 3. Load the model using the absolute path and force local-only mode
        _AI_MODEL_CACHE = SentenceTransformer(model_path, local_files_only=True)
        
    return _AI_MODEL_CACHE
# --- MATPLOTLIB BYPASS: BLIND HOOK FOR ENVIRONMENT STYLING ---
class VisualSafeDataFrame(pd.DataFrame):
    @property
    def style(self):
        class SafeStyler:
            def __init__(self, df): self._df = df
            def __getattr__(self, name):
                def dummy_method(*args, **kwargs): return self
                return dummy_method
            def _repr_html_(self): return self._df.to_html()
            def to_html(self, *args, **kwargs): return self._df.to_html()
        return SafeStyler(self)

class RankingEngine:
    def __init__(self, job_description: str):
        self.job_description = job_description

        # Dual-Engine Architecture 
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # USE THE CACHED MODEL HERE (Fixes the slow startup time)
        self.semantic_model = get_cached_model()

    # --- DYNAMIC EXTRACTORS (Solves the 0.0 Experience Bug) ---
    def _find_exp(self, df):
        """Intelligently hunts down the experience column regardless of its name or nesting."""
        for k in ['experience_years', 'total_experience_years', 'years_of_experience']:
            if k in df.columns: return df[k]
        for c in df.columns:
            if 'exp' in c.lower() and 'year' in c.lower(): return df[c]
        for c in df.columns:
            if isinstance(df[c].dropna().iloc[0] if len(df[c].dropna()) else None, dict):
                def get_e(d):
                    if not isinstance(d, dict): return 0.0
                    for k, v in d.items():
                        if 'exp' in k.lower() and 'year' in k.lower(): return v
                    return 0.0
                res = df[c].apply(get_e)
                if res.sum() > 0: return res
        return pd.Series([0.0] * len(df))

    def _find_title(self, df):
        for k in ['title', 'job_title', 'role', 'current_title', 'designation']:
            if k in df.columns: return df[k]
        for c in df.columns:
            if isinstance(df[c].dropna().iloc[0] if len(df[c].dropna()) else None, dict):
                def get_t(d):
                    if not isinstance(d, dict): return ""
                    for k, v in d.items():
                        if k.lower() in ['title', 'job_title', 'role', 'current_title', 'designation']: return v
                    return ""
                res = df[c].apply(get_t)
                if res.astype(bool).sum() > 0: return res
        return pd.Series([""] * len(df))

    def _find_skills(self, df):
        for k in ['skills', 'core_skills', 'skill_list', 'keywords']:
            if k in df.columns: return df[k]
        for c in df.columns:
            if isinstance(df[c].dropna().iloc[0] if len(df[c].dropna()) else None, dict):
                def get_s(d):
                    if not isinstance(d, dict): return []
                    for k, v in d.items():
                        if 'skill' in k.lower() or 'keyword' in k.lower(): return v
                    return []
                res = df[c].apply(get_s)
                if res.astype(bool).sum() > 0: return res
        return pd.Series([[]] * len(df))

    def process_and_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hybrid Retrieve & Rerank Pipeline: TF-IDF for speed, MiniLM for semantic quality."""
        df = df.copy()
        
        # --- 1. BULLETPROOF DATA EXTRACTION ---
        titles = self._find_title(df).fillna("").astype(str)
        titles_lower = titles.str.lower()
        
        exp_years = pd.to_numeric(self._find_exp(df), errors='coerce').fillna(0.0)
        
        def sanitize_skills(x):
            if isinstance(x, list):
                return [str(i.get('name', i.get('skill', next(iter(i.values())) if i else ""))) if isinstance(i, dict) else str(i) for i in x if i]
            if isinstance(x, str):
                return [s.strip() for s in x.split(",") if s.strip()]
            return []
        
        skills_raw_col = self._find_skills(df)
        clean_skills_list = skills_raw_col.apply(sanitize_skills)
        skills_series = clean_skills_list.apply(lambda x: " ".join(x))
        skills_raw = clean_skills_list.apply(lambda x: x[:3])
        
        if 'redrob_signals' in df.columns: 
            response_rates = df['redrob_signals'].apply(lambda x: float(x.get('recruiter_response_rate', 0.5)) if isinstance(x, dict) else 0.5)
        else: 
            response_rates = pd.Series([0.5] * len(df))

        # --- 2. PHASE 1: FAST LEXICAL FILTER (100,000 -> Top 2,000) ---
        candidate_docs = titles_lower + " " + skills_series.str.lower()
        all_docs = [self.job_description.lower()] + candidate_docs.tolist()
        
        tfidf_matrix = self.tfidf.fit_transform(all_docs)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Phase 1 Trap Application
        bad_titles_pattern = 'marketing|hr|human resources|accountant|sales|content|graphic|recruiter|designer'
        is_bad_title = titles_lower.str.contains(bad_titles_pattern, regex=True, na=False)
        
        penalized_scores = tfidf_scores.copy()
        penalized_scores = np.where(is_bad_title, penalized_scores * 0.01, penalized_scores)
        penalized_scores = np.where((exp_years < 5.0) | (exp_years > 9.0), penalized_scores * 0.1, penalized_scores)
        penalized_scores = np.where((exp_years > 25.0) | (exp_years == 0.0), penalized_scores * 0.001, penalized_scores)
        
        df['temp_score'] = penalized_scores
        
        # Slice out only the top 2000 candidates to pass to the heavy AI model
        id_col = 'candidate_id' if 'candidate_id' in df.columns else df.columns[0]
        top_2000_idx = df.sort_values('temp_score', ascending=False).head(2000).index
        
        # --- 3. PHASE 2: DEEP SEMANTIC RE-RANKING (Top 2,000 -> Top 100) ---
        top_docs = candidate_docs.loc[top_2000_idx].tolist()
        
        # High-Quality AI Inference
        jd_embedding = self.semantic_model.encode([self.job_description])
        candidate_embeddings = self.semantic_model.encode(top_docs)
        semantic_scores = cosine_similarity(jd_embedding, candidate_embeddings).flatten()
        
        final_top_df = df.loc[top_2000_idx].copy()
        
        # Re-apply Behavioral Multipliers and Hacks strictly to the Top 2000
        resp_rates_top = response_rates.loc[top_2000_idx]
        exp_top = exp_years.loc[top_2000_idx]
        is_bad_title_top = is_bad_title.loc[top_2000_idx]
        
        penalties = np.ones(len(final_top_df))
        penalties = np.where(is_bad_title_top, 0.01, penalties)
        penalties = np.where((exp_top < 5.0) | (exp_top > 9.0), penalties * 0.1, penalties)
        penalties = np.where((exp_top > 25.0) | (exp_top == 0.0), penalties * 0.001, penalties)
        
        # Final Score Calculation
        final_scores = (semantic_scores * penalties) * (0.5 + (resp_rates_top * 0.5))
        # Normalize scores so the absolute best candidate is scaled up to ~0.99
        final_scores = (final_scores / final_scores.max()) * 0.9920
        
        # Package for Explanation Engine
        final_top_df['score'] = np.round(final_scores, 4)
        final_top_df['_clean_title'] = titles.loc[top_2000_idx]
        final_top_df['_clean_exp'] = exp_top
        final_top_df['_clean_resp_rate'] = resp_rates_top
        final_top_df['_clean_skills'] = skills_raw.loc[top_2000_idx]
        
        # Extract the absolute top 100
        top_100_raw = final_top_df.sort_values(by=['score', id_col], ascending=[False, True]).head(100).copy()
        top_100_raw['rank'] = range(1, 101)
        
        return VisualSafeDataFrame(top_100_raw)
