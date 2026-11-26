"""Turn free text ("my resume + what I want / never want") into a structured Profile.

Primary path: Groq LLM with JSON output, validated by pydantic.
Fallback path: a simple rule-based extractor, so the app still works without an API key or if the
LLM call fails. The response always says which path produced the profile.
"""
import re

from pydantic import ValidationError

from .llm import GroqClient, LLMError
from .models import SENIORITY_RANK, Profile
from .text import contains_term

SYSTEM = """You extract a structured candidate profile from free text.
Return ONLY a JSON object with these keys:
- skills: list of technical skills the candidate has (lowercase, short names like "python", "react")
- roles: list of job titles the candidate wants
- seniority: one of "intern", "junior", "mid", "senior", or null if unclear
- locations: list of cities the candidate wants to work in (empty if not stated)
- remote_only: true only if the candidate says they need remote work
- min_salary: minimum annual salary in thousands of USD, or null
- dealbreakers: list of short topics/industries/technologies the candidate NEVER wants (e.g. "gambling")
- summary: one sentence describing what the candidate is looking for
Only use information present in the text. Do not invent skills."""

_NEGATION = re.compile(
    r"(?:never|avoid|no|not interested in|don't want|do not want|dont want|won't|refuse)\s+"
    r"(?:want\s+)?(?:to\s+)?(?:work(?:ing)?\s+(?:in|on|with|for)\s+)?(?:in\s+|on\s+|with\s+)?"
    r"([a-z][a-z0-9 /+#.\-]{2,40}?)"
    r"(?=[.,;\n]|$|\sor\s|\sand\s)",
    re.IGNORECASE,
)
_ROLE_WORDS = [
    "ml engineer", "ai engineer", "backend engineer", "frontend engineer", "full-stack developer",
    "data scientist", "data engineer", "data analyst", "devops engineer", "software engineer",
    "mobile developer", "qa engineer", "security analyst", "recommendation systems engineer",
]


def heuristic_profile(text: str, vocab: list[str]) -> Profile:
    lower = text.lower()
    skills = sorted({s for s in vocab if contains_term(lower, s)})

    seniority = None
    if re.search(r"\b(intern|internship|student|final[- ]year|fresher|graduate)\b", lower):
        seniority = "intern"
    elif re.search(r"\b(senior|lead|staff)\b", lower):
        seniority = "senior"
    elif re.search(r"\bjunior\b", lower):
        seniority = "junior"

    dealbreakers = []
    for m in _NEGATION.finditer(text):
        term = m.group(1).strip().lower()
        if term and term not in {"remote", "experience"} and term not in dealbreakers:
            dealbreakers.append(term)

    remote_only = bool(re.search(r"\b(remote only|only remote|must be remote|fully remote)\b", lower))
    roles = [r for r in _ROLE_WORDS if r in lower]
    return Profile(
        skills=skills,
        roles=roles,
        seniority=seniority,
        remote_only=remote_only,
        dealbreakers=dealbreakers,
        summary=" ".join(text.split())[:300],
    )


def _clean(p: Profile) -> Profile:
    p.skills = [s.strip().lower() for s in p.skills if s.strip()]
    p.dealbreakers = [d.strip().lower() for d in p.dealbreakers if d.strip()]
    if p.seniority not in SENIORITY_RANK:
        p.seniority = None
    return p


def extract_profile(text: str, client: GroqClient | None, vocab: list[str]) -> tuple[Profile, str]:
    """Return (profile, source) where source is 'llm' or 'heuristic'."""
    if client is not None and client.available:
        try:
            data = client.chat_json(SYSTEM, text)
            return _clean(Profile.model_validate(data)), "llm"
        except (LLMError, ValidationError):
            pass  # fall through to the rule-based extractor
    return heuristic_profile(text, vocab), "heuristic"
