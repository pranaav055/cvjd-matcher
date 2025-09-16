from flask import Flask, render_template, request, flash
from nlp import tfidf_cosine, present_missing_terms

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-key"  

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cv = (request.form.get("cv") or "").strip()
        jd = (request.form.get("jd") or "").strip()
        try:
            topk = int(request.form.get("topk", 15))
        except ValueError:
            topk = 15
        try:
            ngrams = int(request.form.get("ngrams", 2))
        except ValueError:
            ngrams = 2

        if not cv or not jd:
            flash("Please paste both CV and JD text.")
            return render_template("index.html", result=None, form=request.form)

        ngram_range = (1, ngrams)
        sim = tfidf_cosine(cv, jd, ngram_range=ngram_range)
        present, missing = present_missing_terms(
            cv, jd, ngram_range=ngram_range, top_k=topk
        )

        result = {
            "similarity": round(sim, 4),
            "similarity_pct": int(round(sim * 100)),
            "present": present[:topk],
            "missing": missing[:topk],
            "topk": topk,
            "ngrams": ngrams,
        }
        return render_template("index.html", result=result, form=request.form)

    return render_template("index.html", result=None, form={"topk": 15, "ngrams": 2})

if __name__ == "__main__":
    app.run(debug=True)
