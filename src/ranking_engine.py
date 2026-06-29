import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# --- GLOBAL MODEL CACHE ---
_AI_MODEL_CACHE = None

def get_cached_model():
    """Ensures the heavy AI model is loaded into memory exactly ONCE and from a local, offline path."""
    global _AI_MODEL_CACHE
    if _AI_MODEL_CACHE is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..', 'local_minilm_model')

        if os.path.exists(model_path) and os.path.isdir(model_path) and os.listdir(model_path):
            _AI_MODEL_CACHE = SentenceTransformer(model_path, local_files_only=True)
        else:
            print(f"📥 Downloading MiniLM model to {model_path} for offline caching...")
            _AI_MODEL_CACHE = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            _AI_MODEL_CACHE.save(model_path)
            print(f"✅ Model cached locally at {model_path}")

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
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        self.semantic_model = get_cached_model()

        # Extract JD keywords for explicit matching
        self.jd_keywords = self._extract_jd_keywords(job_description)
        self.jd_required_exp = self._extract_exp_range(job_description)

    # --- JD ANALYSIS ---
    def _extract_jd_keywords(self, jd: str) -> dict:
        """Extract and categorize keywords from JD for explicit matching."""
        jd_lower = jd.lower()

        # Core technical skills (high weight)
        tech_skills = ['python', 'machine learning', 'ml', 'nlp', 'natural language processing',
                       'rag', 'retrieval augmented generation', 'sentence transformer', 'embedding',
                       'vector search', 'pinecone', 'weaviate', 'milvus', 'faiss', 'chromadb',
                       'llm', 'large language model', 'transformer', 'bert', 'pytorch', 'tensorflow',
                       'ranking', 'recommendation', 'search', 'information retrieval', 'semantic search',
                       'fine-tuning', 'lora', 'peft', 'quantization', 'onnx', 'tensorrt',
                       'aws', 'gcp', 'azure', 'docker', 'kubernetes', 'mlops', 'fastapi', 'grpc',
                       'elasticsearch', 'redis', 'kafka', 'spark', 'airflow', 'mlflow']

        # Leadership/ownership signals
        leadership = ['founding', 'founder', 'lead', 'principal', 'architect', 'head of', 'director',
                      'built from scratch', 'end-to-end', 'production', 'scale', 'million', 'billion',
                      'owned', 'designed', 'architected', 'spearheaded', 'launched']

        # Product company signals
        product_signals = ['product company', 'saas', 'b2b', 'consumer', 'platform', 'api', 'sdk',
                          'real users', 'production traffic', 'paying customers', 'revenue']

        found_tech = [kw for kw in tech_skills if kw in jd_lower]
        found_leadership = [kw for kw in leadership if kw in jd_lower]
        found_product = [kw for kw in product_signals if kw in jd_lower]

        return {
            'tech': found_tech,
            'leadership': found_leadership,
            'product': found_product,
            'all': found_tech + found_leadership + found_product
        }

    def _extract_exp_range(self, jd: str) -> tuple:
        """Extract experience range from JD. Returns (min_years, max_years, is_flexible)."""
        jd_lower = jd.lower()
        patterns = [
            r'(\d+)\s*[-–to]\s*(\d+)\s*years?',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\s*[-–to]\s*(\d+)\s*years?',
        ]
        for pat in patterns:
            m = re.search(pat, jd_lower)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    return (int(groups[0]), int(groups[1]), True)
                elif len(groups) == 1:
                    return (int(groups[0]), int(groups[0]) + 4, True)
        return (5, 9, True)  # flexible range from known JD

    # --- DYNAMIC EXTRACTORS (Handles nested dicts in profile/career_history/skills) ---
    def _extract_profile_field(self, df, field_name, default=None):
        """Extract field from profile dict column."""
        if 'profile' not in df.columns:
            return pd.Series([default] * len(df), index=df.index)

        def get_field(p):
            if not isinstance(p, dict):
                return default
            return p.get(field_name, default)

        return df['profile'].apply(get_field)

    def _find_exp(self, df):
        """Extract years of experience from profile.years_of_experience."""
        exp = self._extract_profile_field(df, 'years_of_experience', 0.0)
        return pd.to_numeric(exp, errors='coerce').fillna(0.0)

    def _find_title(self, df):
        """Extract current title from profile.current_title."""
        return self._extract_profile_field(df, 'current_title', '').fillna('').astype(str)

    def _find_skills_detailed(self, df):
        """Extract detailed skills with proficiency and endorsements from skills array."""
        if 'skills' not in df.columns:
            return pd.Series([[] for _ in range(len(df))], index=df.index)

        def parse_skills(skills_list):
            if not isinstance(skills_list, list):
                return []
            parsed = []
            for s in skills_list:
                if isinstance(s, dict):
                    parsed.append({
                        'name': str(s.get('name', s.get('skill', ''))).strip().lower(),
                        'proficiency': s.get('proficiency', 'beginner'),
                        'endorsements': int(s.get('endorsements', 0)),
                        'duration_months': int(s.get('duration_months', 0))
                    })
                elif isinstance(s, str):
                    parsed.append({
                        'name': s.strip().lower(),
                        'proficiency': 'unknown',
                        'endorsements': 0,
                        'duration_months': 0
                    })
            return parsed

        return df['skills'].apply(parse_skills)

    def _get_redrob_signal(self, df, signal_name, default=0.5):
        """Safely extract a redrob_signal field."""
        if 'redrob_signals' not in df.columns:
            return pd.Series([default] * len(df), index=df.index)

        def get_sig(sig):
            if not isinstance(sig, dict):
                return default
            return sig.get(signal_name, default)

        return df['redrob_signals'].apply(get_sig)

    def _compute_skill_match_score(self, candidate_skills, jd_keywords):
        """Compute weighted skill match using proficiency and endorsements."""
        if not candidate_skills:
            return 0.0

        skill_names = [s['name'] for s in candidate_skills]

        # Proficiency weights
        prof_weight = {'expert': 1.0, 'advanced': 0.8, 'intermediate': 0.5, 'beginner': 0.2, 'unknown': 0.3}

        # Direct keyword matches with proficiency weighting
        match_score = 0.0
        for kw in jd_keywords:
            kw_lower = kw.lower()
            for s in candidate_skills:
                if kw_lower in s['name'] or s['name'] in kw_lower:
                    match_score += prof_weight.get(s['proficiency'], 0.3) * (1 + min(s['endorsements'] / 10, 0.5))
                    break

        # Normalize by number of JD keywords
        if jd_keywords:
            match_score = match_score / len(jd_keywords)

        return min(match_score, 1.0)

    def _compute_title_relevance(self, title, jd_keywords):
        """Check if title matches JD expectations (not just keyword stuffing)."""
        title_lower = title.lower()

        # Positive signals (AI/ML engineering, leadership)
        positive_patterns = [
            'ai engineer', 'ml engineer', 'machine learning engineer', 'data scientist',
            'research engineer', 'applied scientist', 'nlp engineer', 'search engineer',
            'ranking engineer', 'recommendation engineer', 'foundation model',
            'ai researcher', 'ml researcher', 'staff engineer', 'principal engineer',
            'founding engineer', 'lead engineer', 'engineering lead', 'tech lead'
        ]

        # Negative signals (trap titles per JD)
        negative_patterns = [
            'marketing', 'hr', 'human resources', 'recruiter', 'sales', 'account',
            'content', 'graphic', 'designer', 'ui', 'ux', 'frontend', 'web developer',
            'full stack', 'backend', 'devops', 'sre', 'data analyst', 'business analyst',
            'product manager', 'project manager', 'scrum', 'agile', 'qa', 'test',
            'support', 'operations', 'admin', 'finance', 'accountant', 'legal'
        ]

        pos_score = sum(1 for p in positive_patterns if p in title_lower)
        neg_score = sum(1 for p in negative_patterns if p in title_lower)

        # If negative signal present, heavily penalize
        if neg_score > 0:
            return -1.0  # Hard trap

        return min(pos_score * 0.2, 1.0)  # Boost for relevant titles

    def _detect_keyword_stuffer(self, row, title_relevance, skill_match):
        """Detect candidates who stuff keywords but have wrong title/background."""
        if skill_match > 0.3 and title_relevance < 0:
            return True
        title = str(row.get('_clean_title', '')).lower()
        if any(t in title for t in ['marketing', 'hr', 'human resources', 'sales', 'recruiter',
                                     'content', 'graphic', 'designer']):
            if skill_match > 0.2:
                return True
        return False

    def _compute_behavioral_score(self, df):
        """Compute composite behavioral signal from redrob_signals (per JD guidance)."""
        signals = {}
        signals['response_rate'] = self._get_redrob_signal(df, 'recruiter_response_rate', 0.5)
        signals['profile_views'] = self._get_redrob_signal(df, 'profile_views_received_30d', 0)
        signals['saved_by_recruiters'] = self._get_redrob_signal(df, 'saved_by_recruiters_30d', 0)
        signals['interview_completion'] = self._get_redrob_signal(df, 'interview_completion_rate', 0.5)
        signals['offer_acceptance'] = self._get_redrob_signal(df, 'offer_acceptance_rate', 0.5)
        signals['github_activity'] = self._get_redrob_signal(df, 'github_activity_score', 0)
        signals['skill_assessment'] = self._get_redrob_signal(df, 'skill_assessment_scores', 0)
        signals['search_appearance'] = self._get_redrob_signal(df, 'search_appearance_30d', 0)
        signals['open_to_work'] = self._get_redrob_signal(df, 'open_to_work_flag', False)
        signals['profile_completeness'] = self._get_redrob_signal(df, 'profile_completeness_score', 50)

        # Normalize count-based signals (log scale)
        for k in ['profile_views', 'saved_by_recruiters', 'search_appearance']:
            signals[k] = signals[k].apply(lambda x: min(np.log1p(x) / 5, 1.0) if x else 0)

        signals['github_activity'] = signals['github_activity'].apply(lambda x: min(x / 100, 1.0))

        def avg_skill_assess(s):
            if isinstance(s, dict) and s:
                return np.mean(list(s.values())) / 100
            return 0.5
        signals['skill_assessment'] = signals['skill_assessment'].apply(avg_skill_assess)

        # Composite behavioral score (weighted per JD emphasis)
        behavioral = (
            0.30 * signals['response_rate'] +
            0.15 * signals['saved_by_recruiters'] +
            0.15 * signals['interview_completion'] +
            0.10 * signals['offer_acceptance'] +
            0.10 * signals['github_activity'] +
            0.10 * signals['skill_assessment'] +
            0.05 * signals['search_appearance'] +
            0.05 * signals['profile_completeness'] / 100
        )

        # Boost if actively looking
        behavioral = behavioral * (1.0 + 0.2 * signals['open_to_work'].astype(float))

        return behavioral.clip(0, 1)

    def _compute_experience_score(self, exp_years, exp_min, exp_max, is_flexible):
        """Compute experience alignment score matching JD philosophy ("range not requirement")."""
        if not is_flexible:
            in_range = (exp_years >= exp_min) & (exp_years <= exp_max)
            return in_range.astype(float)

        score = pd.Series(1.0, index=exp_years.index)

        # Sweet spot: full score
        sweet_spot = (exp_years >= exp_min) & (exp_years <= exp_max)

        # Slightly below: gentle taper
        below = (exp_years >= exp_min - 2) & (exp_years < exp_min)
        score[below] = 0.7 + 0.3 * (exp_years[below] - (exp_min - 2)) / 2

        # Well below: steeper drop
        well_below = (exp_years >= 1) & (exp_years < exp_min - 2)
        score[well_below] = 0.3 * (exp_years[well_below] / max(exp_min - 2, 1))

        # Zero experience: honeypot trap
        score[exp_years == 0] = 0.001

        # Above range: gentle taper for senior candidates
        above = (exp_years > exp_max) & (exp_years <= exp_max + 5)
        score[above] = 0.8 - 0.3 * (exp_years[above] - exp_max) / 5

        # Way above: likely overqualified/wrong level
        way_above = exp_years > exp_max + 5
        score[way_above] = 0.3

        return score.clip(0.001, 1.0)

    def _detect_honeypots(self, df, exp_years, title_relevance, skill_match, behavioral):
        """Detect honeypot candidates per JD guidance."""
        honeypot = pd.Series(False, index=df.index)

        # 1. Zero experience with perfect skills
        honeypot |= ((exp_years == 0) & (skill_match > 0.5))

        # 2. Impossible career trajectory (e.g., 25+ years exp but junior/irrelevant title)
        honeypot |= ((exp_years > 25) & (title_relevance <= 0))

        # 3. Perfect keyword match but zero behavioral signals
        honeypot |= ((skill_match > 0.8) & (behavioral < 0.1))

        # 4. Title mismatch with high skills (keyword stuffer)
        honeypot |= (title_relevance < 0) & (skill_match > 0.3)

        return honeypot

    def process_and_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hybrid Retrieve & Rerank Pipeline aligned with JD requirements."""
        df = df.copy()
        n = len(df)

        # --- 1. EXTRACT ALL FIELDS ---
        exp_years = self._find_exp(df)
        titles = self._find_title(df)
        titles_lower = titles.str.lower()
        skills_detailed = self._find_skills_detailed(df)

        # Skill text for TF-IDF
        skills_text = skills_detailed.apply(lambda lst: ' '.join([s['name'] for s in lst]))

        # JD keywords
        jd_kw = self.jd_keywords['all']
        exp_min, exp_max, exp_flexible = self.jd_required_exp

        # --- 2. COMPUTE SIGNALS FOR ALL CANDIDATES ---
        # Title relevance (trap detection)
        title_relevance = titles.apply(lambda t: self._compute_title_relevance(t, jd_kw))

        # Skill match with proficiency/endorsement weighting
        skill_match = skills_detailed.apply(lambda s: self._compute_skill_match_score(s, jd_kw))

        # Behavioral composite
        behavioral = self._compute_behavioral_score(df)

        # Experience alignment (soft)
        exp_score = self._compute_experience_score(exp_years, exp_min, exp_max, exp_flexible)

        # Honeypot detection
        is_honeypot = self._detect_honeypots(df, exp_years, title_relevance, skill_match, behavioral)

        # --- 3. PHASE 1: FAST LEXICAL FILTER (TF-IDF on title + skills) ---
        candidate_docs = titles_lower + " " + skills_text.str.lower()
        all_docs = [self.job_description.lower()] + candidate_docs.tolist()

        tfidf_matrix = self.tfidf.fit_transform(all_docs)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Apply hard traps early
        phase1_scores = tfidf_scores.copy()
        phase1_scores = np.where(title_relevance < 0, phase1_scores * 0.001, phase1_scores)  # Trap titles
        phase1_scores = np.where(is_honeypot, phase1_scores * 0.0001, phase1_scores)  # Honeypots
        phase1_scores = phase1_scores * (0.3 + 0.7 * exp_score)  # Experience soft gate

        df['phase1_score'] = phase1_scores

        # Top 2000 for semantic reranking
        id_col = 'candidate_id' if 'candidate_id' in df.columns else df.columns[0]
        top_2000_idx = df.sort_values('phase1_score', ascending=False).head(min(2000, n)).index

        # --- 4. PHASE 2: DEEP SEMANTIC RE-RANKING ---
        top_docs = candidate_docs.loc[top_2000_idx].tolist()

        jd_embedding = self.semantic_model.encode([self.job_description])
        candidate_embeddings = self.semantic_model.encode(top_docs, batch_size=32, show_progress_bar=False)
        semantic_scores = cosine_similarity(jd_embedding, candidate_embeddings).flatten()

        # Get signals for top 2000
        top_df = df.loc[top_2000_idx].copy()
        top_titles = titles.loc[top_2000_idx]
        top_exp = exp_years.loc[top_2000_idx]
        top_skill_match = skill_match.loc[top_2000_idx]
        top_title_rel = title_relevance.loc[top_2000_idx]
        top_behavioral = behavioral.loc[top_2000_idx]
        top_honeypot = is_honeypot.loc[top_2000_idx]
        top_skills_raw = skills_detailed.loc[top_2000_idx]

        # Compute final penalties/boosts
        penalties = pd.Series(1.0, index=top_2000_idx)

        # Hard trap: irrelevant title
        penalties = np.where(top_title_rel < 0, 0.001, penalties)

        # Honeypot kill
        penalties = np.where(top_honeypot, 0.0001, penalties)

        # Experience soft gate
        top_exp_score = self._compute_experience_score(top_exp, exp_min, exp_max, exp_flexible)
        penalties = penalties * top_exp_score

        # Behavioral multiplier (0.5 to 1.5)
        behavioral_mult = 0.5 + top_behavioral

        # Skill match bonus (explicit keyword matching bonus)
        skill_bonus = 1.0 + 0.3 * top_skill_match

        # Title relevance bonus
        title_bonus = 1.0 + 0.2 * top_title_rel.clip(0, 1)

        # FINAL SCORE
        final_scores = semantic_scores * penalties * behavioral_mult * skill_bonus * title_bonus

        # Normalize top score to ~0.99
        final_scores = (final_scores / final_scores.max()) * 0.992

        top_df['score'] = np.round(final_scores, 4)
        top_df['_clean_title'] = top_titles
        top_df['_clean_exp'] = top_exp
        top_df['_clean_resp_rate'] = top_behavioral  # Reuse for explanation
        top_df['_clean_skills'] = top_skills_raw.apply(lambda s: [sk['name'] for sk in s[:3]])
        top_df['_skill_match'] = np.round(top_skill_match, 3)
        top_df['_behavioral'] = np.round(top_behavioral, 3)
        top_df['_exp_score'] = np.round(top_exp_score, 3)

        # --- 5. EXTRACT TOP 100 ---
        top_100 = top_df.sort_values(by=['score', id_col], ascending=[False, True]).head(100).copy()
        top_100['rank'] = range(1, 101)

        return VisualSafeDataFrame(top_100)
