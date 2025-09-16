import pytest 
from src.nlp import clean_up_features, tfidf_cosine, present_missing_terms

def test_cleanup():
    terms = [" docker ", "help", "docker", "2025", "git", "g", "building", ""]
    cleaned_terms = clean_up_features(terms)
    assert cleaned_terms == ["docker", "git"]

def test_order():
    cv = "python sql"
    jd = "python python sql"
    present, missing = present_missing_terms(cv, jd, ngram_range=(1,1))
    assert "python" in present and "sql" in present
    i_python = present.index("python")
    i_sql = present.index("sql")
    assert i_python < i_sql
