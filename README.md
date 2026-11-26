# Job Matcher: RAG over live job postings that explains its matches

Paste your skills and what you want (and **never** want). The app turns that into a structured
profile, searches **current job postings fetched from live public feeds**, removes dealbreaker jobs with
hard rules, ranks the rest with hybrid retrieval, and writes a short, checked explanation for each match.
Every result links to the original posting.

**Live demo:** _add your Render URL here after deploying_

## Where the jobs come from

| Source | What it has | Notes |
|---|---|---|
| [Arbeitnow](https://www.arbeitnow.com/blog/job-board-api) | ~300 recent jobs, mostly Europe, includes internships | free API, no key |
| [Remote OK](https://remoteok.com/api) | ~100 remote jobs worldwide | free API, no key |
| [Jobicy](https://jobicy.com/api/v2/remote-jobs) | ~100 remote jobs worldwide | free API, no key |

All three ask for credit and a link to the original posting, so the UI shows the source on every result,
links each result to the original posting (where you apply), and credits the feeds in the footer.
Feeds are fetched in the background on startup and then every 6 hours (not per search) to stay within their
terms. If every feed is down, the app keeps the previous jobs, or the 24 bundled **synthetic** sample jobs
(`data/jobs.json`) if it has never loaded any. The banner on the page says which one you're searching.

## Pipeline

```
live feeds ──► Job objects (cleaned HTML, skills, seniority) ──► search index (swapped in atomically)

free text ──► Profile (Groq LLM → JSON, validated; rule-based fallback)
          ──► hard filters (dealbreakers, seniority, location, salary)   ← rules, not similarity
          ──► hybrid retrieval: BM25 + embeddings, fused with RRF
          ──► relevance gate (must mention enough of your skills, otherwise "no strong matches")
          ──► top-k ──► explanation per job (facts computed in code, LLM only phrases them, output checked)
```

| Decision | Why |
|---|---|
| Dealbreakers are **hard filters** applied before ranking | A similarity score can rank a gambling job highly because the skills match; an LLM can forget an instruction. A "never" must be a rule. |
| BM25 + embeddings, fused with **RRF** | BM25 catches exact terms (`fastapi`), embeddings catch paraphrases. RRF fuses ranks, so no score normalisation is needed. |
| **Relevance gate** after ranking | Similarity always puts *something* first, even when nothing fits (an EE query once ranked an "Account Executive" role first). With 3+ skills a job must mention at least two; otherwise the app says there are no strong matches. |
| Background refresh + **atomic snapshot swap** | Searches never wait on a feed, and never see jobs from one pool with the index of another. |
| Matched skills / gaps computed **in code** | The LLM can't claim a skill the posting doesn't list. |
| LLM explanation is **checked**; falls back to a template | Any number in the text must exist in the posting/candidate text, otherwise the output is discarded (`app/explain.py`). |
| BM25 and cosine written in plain Python | Readable, no numpy needed. |
| Works without an API key | Rule-based profile extractor + template explanations; the UI shows which path was used. |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export GROQ_API_KEY=...                               # PowerShell: $env:GROQ_API_KEY="..."
uvicorn app.main:app --reload
# open http://localhost:8000   (live jobs appear after ~10-30 seconds)
```

Useful switches: `EMBEDDER=hashing` (no model download), `LIVE_JOBS=0` (bundled sample jobs only),
`JOB_REFRESH_HOURS=6`, `MAX_LIVE_JOBS=500`. `GET /api/health` shows the data source and each feed's status.

## Evaluate

```bash
python -m eval.run_eval --embedder hashing     # offline
python -m eval.run_eval --embedder fastembed   # real embeddings (what production uses)
```

The eval runs on the **bundled sample jobs**, not live data, because live data changes every hour and
can't be hand-labelled once. 10 hand-labelled queries (`eval/queries.json`, graded relevance 0-2), metrics
Recall@5, NDCG@5, MRR. Results with the offline hashing embedder (a weak baseline):

| Retrieval | Recall@5 | NDCG@5 | MRR |
|---|---|---|---|
| bm25 | 0.892 | 0.931 | 1.000 |
| dense (hashing) | 0.817 | 0.889 | 1.000 |
| hybrid | 0.833 | 0.912 | 1.000 |

**TODO (you):** run the `fastembed` command and paste that table here. Report what you actually get.

## Limitations (be upfront about these)

- The feeds hold roughly 500 recent jobs, skew senior and European, and have few hardware/EE roles. For a narrow query the honest answer is often "no strong matches". Broader coverage would need another source (for example Adzuna, which has an India endpoint and a free key; not built).
- Remote jobs often have regional restrictions ("USA only"). The location is shown but not filtered on.
- Seniority is inferred from the title, so many postings are "unknown" (they are not filtered out).
- The relevance gate is a heuristic (2 of 3+ skills). It trades recall for precision.
- Eval metrics are on 24 synthetic jobs and 10 queries I labelled; they show the method works, not that it generalises.
- The rule-based fallback only catches the first term in "never X or Y" (the LLM path handles lists).
- In-memory rate limiting per IP; it's a cost guard, not security.
- Free-tier hosting sleeps when idle, and re-embedding ~500 jobs on a small CPU after each wake-up or refresh may take a while (untested on Render's free tier).

## Deploy (Render, free tier)

1. Push this folder to a GitHub repo.
2. Render → New → Blueprint → select the repo (it reads `render.yaml`).
3. In the Render dashboard set `GROQ_API_KEY`. Never commit the key.
4. Open the service URL and paste it at the top of this README.

## Tests

```bash
pytest -q
```
