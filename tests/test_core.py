import pytest

from app import filters
from app.explain import explain, is_grounded, overlap
from app.main import create_app, load_jobs
from app.models import Profile
from app.pipeline import Matcher
from app.retrieval import BM25, HashingEmbedder, reciprocal_rank_fusion
from app.text import contains_term, tokenize


@pytest.fixture(scope="module")
def jobs():
    return load_jobs()


@pytest.fixture(scope="module")
def matcher(jobs):
    return Matcher(jobs, HashingEmbedder(), llm=None)


def test_tokenize_keeps_tech_tokens():
    assert tokenize("C++ and Node.js, CI/CD!") == ["c++", "and", "node.js", "ci", "cd"]


def test_contains_term_is_whole_word():
    assert contains_term("we use javascript", "javascript")
    assert not contains_term("we use javascript", "java")


def test_bm25_prefers_documents_with_rare_query_terms():
    docs = [tokenize(t) for t in ["python web api", "python data science", "kubernetes operators"]]
    scores = BM25(docs).scores(tokenize("kubernetes"))
    assert scores[2] > 0 and scores[0] == 0 and scores[1] == 0


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert max(fused, key=fused.get) == 2  # ranked 2nd and 1st beats 1st and 3rd


def test_dealbreaker_removes_gambling_job(jobs, matcher):
    p = Profile(skills=["python", "django", "payments"], dealbreakers=["gambling"], seniority="junior")
    hits, removed = matcher.rank(p)
    assert "j15" in {job_id for job_id, _ in removed}
    assert "j15" not in {jobs[h.index].id for h in hits}


def test_seniority_filter_excludes_senior_roles_for_interns(jobs):
    p = Profile(seniority="intern")
    allowed, removed = filters.apply(jobs, p)
    assert "j07" in {jid for jid, _ in removed}  # senior role
    assert all(jobs[i].seniority in ("intern", "junior") for i in allowed)


def test_remote_only_filter(jobs):
    allowed, _ = filters.apply(jobs, Profile(remote_only=True))
    assert allowed and all(jobs[i].remote for i in allowed)


def test_overlap_is_deterministic(jobs):
    job = next(j for j in jobs if j.id == "j01")
    matched, gaps = overlap(job, Profile(skills=["Python", "fastapi"]))
    assert matched == ["python", "fastapi"]
    assert set(gaps) == {"llm", "rag", "embeddings"}


def test_grounding_check_rejects_invented_numbers(jobs):
    job = next(j for j in jobs if j.id == "j01")
    p = Profile(skills=["python"])
    assert is_grounded("You already use python for this role.", job, p)
    assert not is_grounded("This role pays 500 and offers 40 days of leave.", job, p)


def test_explain_falls_back_to_template_without_llm(jobs):
    job = jobs[0]
    out = explain(None, Profile(skills=["python"]), [job])
    assert out[0]["reason_source"] == "template" and out[0]["reason"]


def test_pipeline_end_to_end_without_llm(matcher):
    res = matcher.match("Final year student. I know python, rag, llm and fastapi. I never want gambling.", top_k=3)
    assert res["profile_source"] == "heuristic"
    assert 1 <= len(res["results"]) <= 3
    assert "gambling" in res["profile"]["dealbreakers"]
    assert all(r["job"]["id"] != "j15" for r in res["results"])


def test_api_endpoints(matcher):
    from fastapi.testclient import TestClient

    with TestClient(create_app(matcher)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        r = client.post("/api/match", json={"text": "I am a python backend intern who knows fastapi"})
        assert r.status_code == 200 and r.json()["results"]
        assert client.post("/api/match", json={"text": "short"}).status_code == 422
