"""Hard filters: deterministic dealbreakers applied BEFORE any ranking.

Why not let the LLM or the embedding handle "never gambling"? Because a similarity score
can rank a gambling job highly (it matches the skills), and an LLM can forget. A dealbreaker
must be a rule, not a probability.
"""
from .models import SENIORITY_RANK, Job, Profile
from .text import contains_term


def check(job: Job, profile: Profile) -> str | None:
    """Return None if the job passes, otherwise a short human-readable exclusion reason."""
    haystack = job.text()
    for term in profile.dealbreakers:
        if contains_term(haystack, term):
            return f"dealbreaker: mentions '{term}'"

    if profile.seniority in SENIORITY_RANK and job.seniority in SENIORITY_RANK:
        # Allow one level above the candidate (an intern may reasonably apply to junior roles).
        if SENIORITY_RANK[job.seniority] > SENIORITY_RANK[profile.seniority] + 1:
            return f"seniority: {job.seniority} role is above your level"

    if profile.remote_only and not job.remote:
        return "not remote"

    if profile.locations and not job.remote:
        # Real postings write "Paris, Ile-de-France, France"; a wanted "paris" should still match.
        where = job.location.strip().lower()
        wanted = {loc.strip().lower() for loc in profile.locations if loc.strip()}
        if not any(w in where or where in w for w in wanted):
            return f"location: {job.location} is not in your preferred locations"

    if profile.min_salary and job.salary_max is not None and job.salary_max < profile.min_salary:
        return "salary: below your minimum"

    return None


def apply(jobs: list[Job], profile: Profile) -> tuple[list[int], list[tuple[str, str]]]:
    """Return (indices of jobs that pass, [(job_id, reason) for jobs that were removed])."""
    allowed, removed = [], []
    for i, job in enumerate(jobs):
        reason = check(job, profile)
        if reason is None:
            allowed.append(i)
        else:
            removed.append((job.id, reason))
    return allowed, removed
