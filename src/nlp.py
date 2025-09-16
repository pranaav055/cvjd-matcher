from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_cosine(cv_text: str, jd_text: str, ngram_range: Tuple[int, int] = (1,2)) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)
    X = vec.fit_transform([cv_text, jd_text])
    score = cosine_similarity(X[0], X[1])[0,0]
    return float(score)
