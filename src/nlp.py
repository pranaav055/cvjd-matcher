from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

junk = {
    "help","responsibilities","intern","role","write","looking","plus",
    "ability","work","requirements","we","need","needs","good","strong",
    "communication","collaborate","build","building"
}

def clean_up_features(terms):
    out, seen = [], set()
    for t in terms:
        if not t:
            continue
        t = t.strip()
        if t in seen:
            continue
        elif any(c.isdigit() for c in t):
            continue
        elif len(t) <= 2:
            continue
        elif t in junk:
            continue
        out.append(t)
        seen.add(t)
    return out


def tfidf_cosine(cv_text: str, jd_text: str, ngram_range: Tuple[int, int] = (1,2)) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)
    X = vec.fit_transform([cv_text, jd_text])
    score = cosine_similarity(X[0], X[1])[0,0]
    return float(score)

def present_missing_terms(cv_text:str, jd_text: str, ngram_range: Tuple[int, int] = (1,2), top_k: int = 15) -> Tuple[List[str], List[str]]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)
    X = vec.fit_transform([cv_text, jd_text])
    
    # get the vocabulary and cv and jd vectors 
    vocab = vec.get_feature_names_out()
    cv_vec = X[0].toarray().ravel()
    jd_vec = X[1].toarray().ravel()

    # array of T,F for empty and non epmty tfidf values
    cv_mask = cv_vec > 0
    jd_mask = jd_vec > 0

    # boolean masks for tokesn or features which are present in both cv and jd
    present_mask = cv_mask & jd_mask
    # boolean mask for toeksn or features not present in cv but present in jd
    not_present_mask = jd_mask & ~cv_mask

    # gives the array of indexs for which the tfidf values are present/freatures are present in both 
    present_index = np.where(present_mask)[0]
    # gives the array of indexs for which features present in jd but not in cv
    not_present_index = np.where(not_present_mask)[0]

     # add the tfidf values of the features which are present in both
    cv_present_values = []
    for i in present_index:
        cv_present_values.append(jd_vec[i])

    # add the tfidf values of the features which are not present in both
    jd_not_present_values = []
    for i in not_present_index:
        jd_not_present_values.append(jd_vec[i])
    
    # store the indexs of sorted tfidf jd values from the sliced array, these indexs dont match to the vocab arrays indexs
    not_present_index_local = np.array([])
    not_present_index_local = np.argsort(-np.array(jd_not_present_values))

    # store the indexs of sorted tfidf jd values from the sliced array, these indexs dont match to the vocab arrays indexs
    present_index_local = np.array([])
    present_index_local = np.argsort(-np.array(cv_present_values))

    # store the indexs of sorted tfidf jd values, which coresppond to the vocab array
    not_present_index_global = []
    for i in range(len(not_present_index_local)):
        not_present_index_global.append(not_present_index[not_present_index_local[i]])

     # store the indexs of sorted tfidf cv values, which coresppond to the vocab array
    present_index_global = []
    for i in range(len(present_index_local)):
        present_index_global.append(present_index[present_index_local[i]])
    
    # add features/terms which are not present in both
    not_present = []
    for i in not_present_index_global[:top_k]:
        not_present.append(vocab[i])
    
    # add features/terms which are present in both
    present = []
    for i in present_index_global:
        present.append(vocab[i])

    present = clean_up_features(present)
    not_present = clean_up_features(not_present)

    return present, not_present