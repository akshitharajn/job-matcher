# Job Matcher: RAG that explains its matches

Paste your skills and what you want (and **never** want). The app turns that into a structured
profile, removes dealbreaker jobs with hard rules, ranks the rest with hybrid retrieval, and writes
a short, checked explanation for each match.

> Demo data: 24 **synthetic** job postings in `data/jobs.json`. No real companies or people.

**Live demo:** _add your Render URL here after deploying_

## Pipeline

```
free text ──► Profile (Groq LLM → JSON, validated; rule-based fallback)
          ──► hard filters (dealbreakers, seniority, location, salary)   ← rules, not similarity
          ──► hybrid retrieval: BM25 + embeddings, fused with RRF
          ──► top-k ──► explanation per job (facts computed in code, LLM only phrases them, output checked)
```

Key design decisions (each is something to be able to defend):

| Decision | Why |
|---|---|
| Dealbreakers are **hard filters** applied before ranking | A similarity score can rank a gambling job highly because the skills match; an LLM can forget an instruction. A "never" must be a rule. |
| BM25 + embeddings, fused with **RRF** | BM25 catches exact terms (`fastapi`), embeddings catch paraphrases ("suggest movies" → recommender). RRF fuses ranks, so no score normalisation is needed. |
| Matched skills / gaps computed **in code** | The LLM can't claim a skill the posting doesn't list. |
| LLM explanation is **checked**; falls back to a template | Any number in the text must exist in the posting/candidate text, otherwise the output is discarded (`app/explain.py`). |
| BM25 and cosine written in plain Python | Small corpus, readable, no numpy needed. |
| Works without an API key | Rule-based profile extractor + template explanations; the UI shows which path was used. |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                  # then put your Groq key in .env
export $(grep -v '^#' .env | xargs)                   # or set the variables however you prefer
uvicorn app.main:app --reload
# open http://localhost:8000
```

Offline (no model download, no key): `EMBEDDER=hashing uvicorn app.main:app`

## Evaluate

```bash
python -m eval.run_eval --embedder hashing     # offline
python -m eval.run_eval --embedder fastembed   # real embeddings (what production uses)
```

10 hand-labelled queries (`eval/queries.json`, graded relevance 0-2), metrics Recall@5, NDCG@5, MRR.

Results with the offline hashing embedder (a weak baseline with no semantic understanding):

| Retrieval | Recall@5 | NDCG@5 | MRR |
|---|---|---|---|
| bm25 | 0.892 | 0.931 | 1.000 |
| dense (hashing) | 0.817 | 0.889 | 1.000 |
| hybrid | 0.833 | 0.912 | 1.000 |

**TODO (you):** run the `fastembed` command above and paste that table here. Expect the picture to
change, since real embeddings handle paraphrases the hashing baseline can't. Report whatever you
actually get, even if BM25 still wins on a corpus this small.

## Limitations (be upfront about these)

- 24 synthetic postings and 10 queries: the metrics show the method works, they don't prove it generalises.
- Eval labels were drafted from the job texts, not by independent annotators. Review and extend them.
- The rule-based fallback only catches the first term in "never X or Y" (the LLM path handles lists).
- Seniority/location/salary filters rely on the LLM extracting them correctly.
- In-memory rate limiting per IP; it's a cost guard, not security.
- Free-tier hosting sleeps when idle, so the first request after a pause is slow.

## Deploy (Render, free tier)

1. Push this folder to a GitHub repo.
2. Render → New → Blueprint → select the repo (it reads `render.yaml`).
3. In the Render dashboard set `GROQ_API_KEY`. Never commit the key.
4. Open the service URL and paste it at the top of this README.

## Tests

```bash
pytest -q
```
