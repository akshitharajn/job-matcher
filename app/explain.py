"""Grounded match explanations.

Design: facts are computed in CODE (skill overlap, gaps, filters). The LLM only turns those facts
into a readable sentence, and its output is checked: any number it mentions must exist in the
posting or the candidate text, otherwise we discard it and use a template. This limits
hallucination instead of hoping the prompt prevents it.
"""
import json
import re

from .llm import GroqClient, LLMError
from .models import Job, Profile
from .text import contains_term

SYSTEM = """You write short, factual explanations of why a job fits a candidate.
Use ONLY the facts provided. Never invent skills, salary, benefits or company details.
For each job write 1-2 sentences addressed to the candidate ("you"). Mention the matched skills,
and if there are gaps, mention one honestly.
Return ONLY JSON: {"reasons": {"<job_id>": "<text>", ...}}"""


def overlap(job: Job, profile: Profile) -> tuple[list[str], list[str]]:
    """(matched skills, gap skills) - computed deterministically."""
    have = {s.lower() for s in profile.skills}
    matched, gaps = [], []
    for skill in job.skills:
        if skill.lower() in have or contains_term(profile.summary, skill):
            matched.append(skill)
        else:
            gaps.append(skill)
    return matched, gaps


def template_reason(job: Job, matched: list[str], gaps: list[str]) -> str:
    where = "remote" if job.remote else job.location
    if matched:
        s = f"You already use {', '.join(matched)}, which the {job.title} role at {job.company} ({where}) asks for."
    else:
        s = f"The {job.title} role at {job.company} ({where}) is the closest available fit to what you described."
    if gaps:
        s += f" You would need to pick up: {', '.join(gaps)}."
    return s


def is_grounded(reason: str, job: Job, profile: Profile) -> bool:
    """Reject explanations that contain numbers absent from the posting or the candidate text."""
    source = f"{job.text()} {job.salary_min} {job.salary_max} {profile.summary} {profile.min_salary}"
    return all(n in source for n in re.findall(r"\d[\d,.]*\d|\d", reason))


def explain(client: GroqClient | None, profile: Profile, jobs: list[Job]) -> list[dict]:
    """Return [{matched_skills, gaps, reason, reason_source}] aligned with `jobs`."""
    facts = []
    for job in jobs:
        matched, gaps = overlap(job, profile)
        facts.append((matched, gaps))

    llm_reasons: dict[str, str] = {}
    if client is not None and client.available and jobs:
        payload = {
            "candidate": profile.summary,
            "jobs": [
                {
                    "job_id": j.id,
                    "title": j.title,
                    "company": j.company,
                    "seniority": j.seniority,
                    "location": "remote" if j.remote else j.location,
                    "matched_skills": m,
                    "missing_skills": g,
                    "posting": j.description,
                }
                for j, (m, g) in zip(jobs, facts)
            ],
        }
        try:
            data = client.chat_json(SYSTEM, json.dumps(payload))
            llm_reasons = {k: v for k, v in data.get("reasons", {}).items() if isinstance(v, str)}
        except LLMError:
            llm_reasons = {}

    out = []
    for job, (matched, gaps) in zip(jobs, facts):
        reason, source = llm_reasons.get(job.id), "llm"
        if not reason or not is_grounded(reason, job, profile):
            reason, source = template_reason(job, matched, gaps), "template"
        out.append({"matched_skills": matched, "gaps": gaps, "reason": reason, "reason_source": source})
    return out
