"""Live-feed code, tested with small hand-written payloads shaped like the real feeds (no network)."""
import pytest

from app import filters
from app.main import create_app, load_jobs
from app.models import Job, Profile
from app.pipeline import Matcher
from app.providers import (clean_html, infer_seniority, parse_arbeitnow, parse_jobicy, parse_remoteok,
                           safe_url)
from app.retrieval import HashingEmbedder
from app.store import JobStore, merge_jobs

ARBEITNOW = {"data": [
    {"slug": "ai-engineer-intern-paris-1", "company_name": "Acme SAS", "title": "AI Engineer Intern (2027)",
     "description": "<p>Build <b>RAG</b> pipelines in Python &amp; FastAPI.</p><script>x()</script>",
     "remote": False, "url": "https://www.arbeitnow.com/jobs/companies/acme/ai-engineer-intern-paris-1",
     "tags": ["Engineering"], "job_types": ["internship"], "location": "Paris, France", "created_at": 1795692671},
    {"slug": "broken", "company_name": "No Title Co"},  # malformed: must be skipped, not crash
]}
REMOTEOK = [
    {"last_updated": 1, "legal": "API Terms of Service: link back"},
    {"id": "11", "position": "Senior Python Developer", "company": "Globex", "tags": ["python", "backend"],
     "description": "<div>Django and PostgreSQL</div>", "location": "Worldwide", "salary_min": 90000,
     "salary_max": 120000, "date": "2026-11-25T13:00:35+00:00", "url": "https://remoteOK.com/remote-jobs/11"},
    {"id": "12", "position": "Marketing Lead", "company": "Initech", "tags": ["marketing"], "description": "",
     "salary_min": 0, "salary_max": 0, "url": "javascript:alert(1)"},
]
JOBICY = {"jobs": [
    {"id": 5, "url": "https://jobicy.com/jobs/5-junior-data-analyst", "jobTitle": "Junior Data Analyst",
     "companyName": "Hooli", "jobGeo": "Brazil,  Mexico", "jobLevel": "Junior",
     "jobDescription": "<p>SQL and Tableau dashboards</p>", "pubDate": "2026-11-26T07:45:10+00:00"},
]}


def test_clean_html_strips_tags_scripts_and_entities():
    assert clean_html("<p>A &amp; B</p><script>evil()</script><style>x{}</style>") == "A & B"


def test_safe_url_only_allows_http_links():
    assert safe_url("https://x.example/a") and safe_url("http://x.example")
    assert safe_url("javascript:alert(1)") is None and safe_url(None) is None


@pytest.mark.parametrize("title,hint,expected", [
    ("AI Engineer Intern (2027)", "", "intern"), ("Praktikant:in New Business", "", "intern"),
    ("Senior Python Developer", "", "senior"), ("Junior Data Analyst", "", "junior"),
    ("Software Engineer", "Midweight", "mid"), ("Software Engineer", "", "unknown"),
])
def test_infer_seniority(title, hint, expected):
    assert infer_seniority(title, hint) == expected


def test_parse_arbeitnow():
    jobs = parse_arbeitnow(ARBEITNOW)
    assert len(jobs) == 1  # the malformed item was skipped
    j = jobs[0]
    assert (j.source, j.seniority, j.remote, j.posted) == ("Arbeitnow", "intern", False, "2026-11-26")
    assert "evil" not in j.description and "RAG pipelines in Python & FastAPI" in j.description
    assert {"python", "fastapi", "rag"} <= set(j.skills) and j.url.startswith("https://www.arbeitnow.com/")


def test_parse_remoteok_skips_legal_notice_and_unsafe_urls():
    jobs = parse_remoteok(REMOTEOK)
    assert [j.id for j in jobs] == ["ro-11", "ro-12"]
    a, b = jobs
    assert (a.seniority, a.salary_min, a.salary_max, a.remote, a.source) == ("senior", 90, 120, True, "Remote OK")
    assert b.salary_min is None and b.url is None  # 0 salary = unknown; javascript: link dropped


def test_parse_jobicy():
    (j,) = parse_jobicy(JOBICY)
    assert (j.seniority, j.location, j.source) == ("junior", "Brazil, Mexico", "Jobicy")
    assert {"sql", "tableau"} <= set(j.skills)


def test_location_filter_matches_substrings():
    job = Job(id="x", title="t", company="c", location="Paris, Ile-de-France, France", remote=False,
              seniority="unknown", skills=[], description="")
    assert filters.check(job, Profile(locations=["Paris"])) is None
    assert filters.check(job, Profile(locations=["Berlin"])) is not None


def test_unknown_seniority_is_not_filtered_out():
    job = Job(id="x", title="t", company="c", location="x", remote=True, seniority="unknown", skills=[], description="")
    assert filters.check(job, Profile(seniority="intern")) is None


def _job(i, title, company):
    return Job(id=str(i), title=title, company=company, location="x", remote=True, seniority="unknown",
               skills=[], description="")


def test_merge_jobs_interleaves_dedupes_and_caps():
    a = [_job(1, "Dev", "A"), _job(2, "Dev", "B"), _job(3, "Dev", "C")]
    b = [_job(4, "dev", "a"), _job(5, "Ops", "D")]  # first one duplicates a[0] ignoring case
    merged = merge_jobs([a, b], cap=10)
    assert [j.id for j in merged] == ["1", "2", "5", "3"]
    assert len(merge_jobs([a, b], cap=2)) == 2


class FakeProvider:
    def __init__(self, name, jobs=None, error=None):
        self.name, self.jobs, self.error = name, jobs or [], error

    def fetch(self):
        if self.error:
            raise self.error
        return self.jobs


def _matcher():
    return Matcher(load_jobs(), HashingEmbedder(), llm=None)


def test_store_swaps_in_live_jobs_and_search_uses_them():
    m = _matcher()
    assert m.snapshot.source == "sample"
    live = parse_arbeitnow(ARBEITNOW) + parse_remoteok(REMOTEOK)
    store = JobStore(m, [FakeProvider("Arbeitnow", live[:1]), FakeProvider("Remote OK", live[1:])])
    status = store.refresh()
    assert status["state"] == "live" and m.snapshot.source == "live" and len(m.jobs) == 3
    res = m.match("I know python and fastapi and rag, looking for an AI engineer internship in Paris")
    assert res["data"]["source"] == "live"
    assert res["results"][0]["job"]["source"] == "Arbeitnow"
    assert res["results"][0]["job"]["url"].startswith("https://www.arbeitnow.com/")


def test_store_keeps_previous_jobs_when_all_feeds_fail():
    m = _matcher()
    store = JobStore(m, [FakeProvider("A", error=RuntimeError("down")), FakeProvider("B", error=ValueError("bad"))])
    status = store.refresh()
    assert m.snapshot.source == "sample" and len(m.jobs) == 24
    assert "failed" in status["state"] and set(status["errors"]) == {"A", "B"}


def test_store_survives_one_failing_feed():
    m = _matcher()
    store = JobStore(m, [FakeProvider("A", error=RuntimeError()), FakeProvider("B", parse_remoteok(REMOTEOK))])
    status = store.refresh()
    assert m.snapshot.source == "live" and status["errors"] == {"A": "RuntimeError"}


def test_health_endpoint_reports_data_source():
    from fastapi.testclient import TestClient

    m = _matcher()
    store = JobStore(m, [FakeProvider("A", parse_remoteok(REMOTEOK))])
    store.refresh()
    with TestClient(create_app(m)) as client:  # matcher supplied => no background thread started
        body = client.get("/api/health").json()
        assert body["data_source"] == "live" and body["jobs"] == 2 and body["feeds"] is None


def test_category_tags_are_not_treated_as_skills():
    from app.skills import extract_skills

    skills = extract_skills(["Data Scientist", "Engineering", "python", "remote"], "We use django daily")
    assert "data scientist" not in skills and "engineering" not in skills and "remote" not in skills
    assert skills[:2] == ["python", "django"]
