# CV ↔ JD Matcher (TF-IDF + Cosine)

Compares CV and Job descriptions to output back:
- Overall similarity score and percentage
- 0.0~0.3 - unrelated, 0.04~0.12 - loosely related, 0.13~0.25 - moderately related, 0.26+ - strongly related
- Present terms both in CV and JD
- Missing terms in JD, ranked by importance (TFIDF values in JD vector)

Built to be used for quick internship/job targeting

## Features 
- Used NLP: Made CV and JD into TF-IDF vectors + used cosine similarity to compare two files, can change n-grams to uni/bi//tri
- Gives back terms which are only in JD: terms are ranked based on TF-IDF values for the JD vector, hence can add these skill into CV based on what is of more value
- Two interfaces:
    - CLI: python -m src.cli --cv <cv.txt> --jd <jd.txt> --ngrams 1 --topk 15
    - Web UI (Flask): paste CV & JD, hit Match 
- Quality/Testing: wrote unit tests using pytest, maintained clean repo structure.

## Tech stack
Python, scikit-learn, Flask, NumPy, pytest

## Setup 
1 - Clone repo
git clone https://github.com/pranaav055/cvjd-matcher
cd cvjd-matcher

2 - Set up venv
python -m venv .venv
.venv\Scripts\activate    # Windows
#source .venv/bin/activate  #Mac/Linux

3 - Install dependencies 
pip install -r requirements.txt

## Run the CLI
input: python -m src.cli --cv samples/cv.txt --jd samples/jd_software_intern.txt --topk 10 --ngrams 2

output: {
  "similarity score": 0.1059, 
  "similarity percentage": 11,
  "present terms/skills": [   
    "apis",
    "rest apis",
    "rest",
    "python",
    "cloud deployment",       
    "clean",
    "cloud",
    "docker cloud",
    "docker",
    "deployment"
  ],
  "missing terms/skills": [   
    "code",
    "apis responsibilities",  
    "apis write",
    "aws",
    "aws ci",
    "big",
    "big plus"
  ]
}

flags:
--topk -> how many terms to show
--ngrams -> max ngram size

## Run Web UI
input:
python -m src.web
- paste CV and JD text chose top K and N-grams and press match 
- Screenshots in static folder 

## How it works
1) Vectorise CV and JD with TfidfVectorizer
    - English stop words to not use common english words, lowercasing, configurable ngram_range
2) Similarity: cosine between the two vectors to find how similar each vector is (range 0~1)
3) Present terms: includes the features/terms which have TF-IDF values > 0 for both CV and JD vectors, the terms are in descending order by the TF-IDF value of that term in the JD vector 
4) Missing terms: includes the features/terms which have TF-IDF value > 0 in JD vector but 0 in CV vector, the terms are in descending order by the TF-IDF value of tjat term in the JD vector 

Cosine similarity does not mean that documents are going to be a perfect match. Below are typical scores:

0.00–0.03 unrelated
0.04–0.12 loosely related
0.13–0.25 moderately aligned
0.26+ strongly aligned
(Depends on doc length and n-gram config.)

## Project structure 

src/
  nlp.py          # tfidf_cosine, present_missing_terms, cleaners
  textio.py       # load_text helper
  cli.py          # command-line interface
  web.py          # Flask app
  templates/
    index.html
  static/
    style.css
tests/
  test_textio.py
  test_nlp_similarity.py
  test_present_not_present.py
  test_clean_order.py
samples/
  cv.txt
  jd_software_intern.txt
  jd_data_intern.txt
  jd_far.txt

## Run unit test
pytest -q




