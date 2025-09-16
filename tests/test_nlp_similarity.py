import pytest
from src.nlp import tfidf_cosine

def test_tfidf_cosine_base_case():
    cv = "python machine learing and javascript and good comunication"
    jd_close = "need python javascript and comunication"
    jd_far = "baking skills good whisking and early timing"
    s_close = tfidf_cosine(cv, jd_close)
    s_far = tfidf_cosine(cv, jd_far)
    assert (0 <= s_close <=1) and (0 <= s_far <=1)
    assert s_close > s_far

def test_tfidf_cosine_same():
    cv = "python machine learing and javascript and good comunication"
    jd = "python machine learing and javascript and good comunication"
    s = tfidf_cosine(cv, jd)
    assert s == pytest.approx(1.0, rel=1e-12)