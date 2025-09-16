import argparse
import json
import sys
from src.textio import load_text
from src.nlp import tfidf_cosine, present_missing_terms

def main():
    ap = argparse.ArgumentParser(description="A CV to Job description matcher which gives back a similarity score and missing terms")
    ap.add_argument("--cv", required=True, help="Path to the cv file (.txt)")
    ap.add_argument("--jd", required=True, help="Path to the jd file (.txt)")
    ap.add_argument("--ngrams", type=int, default=2, choices=[1, 2, 3], help="N grams of vocabulary of corpus wanted for comparison of cv and jd 1: Unigrms, 2: Unigrams to Bigrams, 3: Unigrams to Trigrams")
    ap.add_argument("--topk", type=int, default=15, help="Number of missing terms wanted to be returned")

    args = ap.parse_args()

    ngram_range = (1, args.ngrams)

    try:
        cv_text = load_text(args.cv)
        jd_text = load_text(args.jd)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(2)
    s = tfidf_cosine(cv_text, jd_text, ngram_range=ngram_range)
    present, missing = present_missing_terms(cv_text, jd_text, ngram_range=ngram_range, top_k=args.topk)

    out = {
        "similarity score": round(s, 4),
        "similarity percentage": int(round(s * 100)),
        "present terms/skills": present[:args.topk],
        "missing terms/skills": missing
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
