"""Live job feeds -> Job objects.

Sources (all free public APIs, no key needed):
  Arbeitnow  https://www.arbeitnow.com/blog/job-board-api   (mostly Europe, includes internships)
  Remote OK  https://remoteok.com/api                        (remote jobs worldwide)
  Jobicy     https://jobicy.com/api/v2/remote-jobs           (remote jobs worldwide)

Usage terms, as stated by each feed: credit the source and link to the original posting; do not
abuse the API. This app therefore (1) shows the source on every result, (2) links every result to the
original posting, and (3) only refreshes a few times a day (see store.py).
"""
import html
import re
from datetime import datetime, timezone

import httpx

from .models import Job
from .skills import extract_skills

USER_AGENT = "job-matcher-portfolio-demo (non-commercial)"
MAX_DESC = 1200


# ---------------------------------------------------------------- helpers
def clean_html(raw: str | None, limit: int = MAX_DESC) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())[:limit]


def safe_url(url) -> str | None:
    """Only http(s) links are ever rendered as links."""
    return url if isinstance(url, str) and url.startswith(("https://", "http://")) else None


def iso_from_epoch(value) -> str | None:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def infer_seniority(title: str, hint: str = "") -> str:
    t = f"{title} {hint}".lower()
    if re.search(r"\b(intern|internship|praktik\w*|stage|stagiaire|trainee|werkstudent\w*|working student|apprentice)\b", t):
        return "intern"
    if re.search(r"\b(senior|sr|lead|principal|staff|head|director|manager|vp)\b", t):
        return "senior"
    if re.search(r"\b(junior|jr|graduate|entry[- ]level|associate)\b", t):
        return "junior"
    if re.search(r"\b(mid|mid-level|midweight|intermediate)\b", t):
        return "mid"
    return "unknown"  # unknown levels are not filtered out; ranking decides


def _salary_k(value) -> int | None:
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    return v // 1000 if v > 0 else None


# ---------------------------------------------------------------- parsers (pure functions, unit-tested)
def parse_arbeitnow(payload: dict) -> list[Job]:
    jobs = []
    for it in payload.get("data", []):
        try:
            title, desc = it["title"], clean_html(it.get("description"))
            jobs.append(Job(
                id=f"an-{it['slug']}", title=title, company=it.get("company_name") or "Unknown",
                location=it.get("location") or "Not specified", remote=bool(it.get("remote")),
                seniority=infer_seniority(title, " ".join(it.get("job_types") or [])),
                skills=extract_skills(it.get("tags"), f"{title} {desc}"),
                description=desc, url=safe_url(it.get("url")), source="Arbeitnow",
                posted=iso_from_epoch(it.get("created_at")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return jobs


def parse_remoteok(payload: list) -> list[Job]:
    jobs = []
    for it in payload:
        if not isinstance(it, dict) or "position" not in it:  # first element is the legal notice
            continue
        try:
            title, desc = it["position"], clean_html(it.get("description"))
            jobs.append(Job(
                id=f"ro-{it['id']}", title=title, company=it.get("company") or "Unknown",
                location=it.get("location") or "Remote", remote=True,
                seniority=infer_seniority(title), salary_min=_salary_k(it.get("salary_min")),
                salary_max=_salary_k(it.get("salary_max")),
                skills=extract_skills(it.get("tags"), f"{title} {desc}"), description=desc,
                url=safe_url(it.get("url") or it.get("apply_url")), source="Remote OK",
                posted=(it.get("date") or "")[:10] or None,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return jobs


def parse_jobicy(payload: dict) -> list[Job]:
    jobs = []
    for it in payload.get("jobs", []):
        try:
            title = it["jobTitle"]
            desc = clean_html(it.get("jobDescription") or it.get("jobExcerpt"))
            geo = it.get("jobGeo")
            location = ", ".join(g.strip() for g in geo.split(",") if g.strip()) if isinstance(geo, str) else "Remote"
            level = it.get("jobLevel") or ""
            jobs.append(Job(
                id=f"jb-{it['id']}", title=title, company=it.get("companyName") or "Unknown",
                location=location or "Remote", remote=True,
                seniority=infer_seniority(title, level if level.lower() not in ("any", "") else ""),
                skills=extract_skills(None, f"{title} {desc}"), description=desc,
                url=safe_url(it.get("url")), source="Jobicy", posted=(it.get("pubDate") or "")[:10] or None,
            ))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return jobs


# ---------------------------------------------------------------- fetchers
def _get_json(url: str, params: dict | None = None, timeout: float = 25.0):
    r = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.json()


class Provider:
    name = ""
    credit_url = ""

    def fetch(self) -> list[Job]:
        raise NotImplementedError


class ArbeitnowProvider(Provider):
    name, credit_url = "Arbeitnow", "https://www.arbeitnow.com"

    def fetch(self) -> list[Job]:
        return parse_arbeitnow(_get_json("https://www.arbeitnow.com/api/job-board-api", {"page": 1}))


class RemoteOKProvider(Provider):
    name, credit_url = "Remote OK", "https://remoteok.com"

    def fetch(self) -> list[Job]:
        return parse_remoteok(_get_json("https://remoteok.com/api"))


class JobicyProvider(Provider):
    name, credit_url = "Jobicy", "https://jobicy.com"

    def fetch(self) -> list[Job]:
        return parse_jobicy(_get_json("https://jobicy.com/api/v2/remote-jobs", {"count": 100}))


def default_providers() -> list[Provider]:
    return [ArbeitnowProvider(), RemoteOKProvider(), JobicyProvider()]
