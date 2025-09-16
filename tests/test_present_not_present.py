import pytest
from src.nlp import present_missing_terms

def test_present_missing_base():
    cv = "skills: python, cpp, web ui, deployment, unit testing, GIT"
    jd = "Requirments: python, java, cpp, machine learning, web ui, low level architeture, deployment, technical ability, communication, SQL, GIT, unit testing"
    present, missing = present_missing_terms(cv, jd, top_k=10)
    assert "python" in present
    assert (("web ui" in present) or ("web" in present and "ui" in present))
    assert any((m == "technical ability") or ("technical" in m) for m in missing)
    assert len(missing) <= 10