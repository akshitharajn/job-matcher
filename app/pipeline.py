"""The full matching pipeline.

free text -> Profile (LLM) -> hard filters -> hybrid retrieval -> top-k -> grounded explanations

The job pool lives in an immutable Snapshot. A refresh builds a whole new Snapshot and swaps one
reference, so a search never sees jobs from one pool and an index from another.
"""
from dataclasses import dataclass, field

from . import filters
from .explain import explain
from .llm import GroqClient
from .models import Job, Profile
from .profile import extract_profile
from .retrieval import Embedder, Hit, HybridIndex
from .skills import GENERIC_TAGS, SKILL_VOCAB
from .text import contains_term

MAX_FILTER_REPORT = 20  # how many removed jobs to list in the response (the rest are only counted)


def mentions_a_skill(job: Job, profile: Profile) -> bool:
    """Relevance gate: similarity scores always rank *something* first, even when nothing fits.
    With 1-2 skills a job must mention one of them; with 3+ it must mention at least two, because a
    single hit is often an ordinary word ("embedded in the team", "networking events")."""
    skills = [s for s in profile.skills if s.lower() not in GENERIC_TAGS and len(s) > 1]
    if not skills:
        return True  # nothing to check against; fall back to ranking alone
    text = job.text()
    needed = 1 if len(skills) <= 2 else 2
    return sum(contains_term(text, s) for s in skills) >= needed


@dataclass(frozen=True)
class Snapshot:
    jobs: list[Job]
    index: HybridIndex
    vocab: list[str]
    source: str  # "sample" or "live"
    info: dict = field(default_factory=dict)


class Matcher:
    def __init__(self, jobs: list[Job], embedder: Embedder, llm: GroqClient | None = None,
                 source: str = "sample", info: dict | None = None):
        self.embedder = embedder
        self.llm = llm
        self._snap = self._build(jobs, source, info or {})

    def _build(self, jobs: list[Job], source: str, info: dict) -> Snapshot:
        # Vocabulary for the rule-based extractor: known skills + any skill seen in at least 2 jobs.
        counts: dict[str, int] = {}
        for j in jobs:
            for s in {s.lower() for s in j.skills}:
                counts[s] = counts.get(s, 0) + 1
        vocab = sorted({*SKILL_VOCAB, *(s for s, n in counts.items() if n >= 2 and s not in GENERIC_TAGS)})
        return Snapshot(jobs, HybridIndex(jobs, self.embedder), vocab, source, info)

    def set_jobs(self, jobs: list[Job], source: str, info: dict | None = None) -> None:
        self._snap = self._build(jobs, source, info or {})  # build first, then swap one reference

    @property
    def snapshot(self) -> Snapshot:
        return self._snap

    @property
    def jobs(self) -> list[Job]:
        return self._snap.jobs

    def rank(self, profile: Profile, mode: str = "hybrid", snap: Snapshot | None = None):
        """Filters first, then retrieval over the jobs that survived. Used by the API and the evals."""
        snap = snap or self._snap
        allowed, removed = filters.apply(snap.jobs, profile)
        return snap.index.search(profile.query_text(), mode=mode, allowed=allowed), removed

    def match(self, text: str, top_k: int = 5, mode: str = "hybrid") -> dict:
        snap = self._snap  # one consistent view for the whole request
        profile, profile_source = extract_profile(text, self.llm, snap.vocab)
        hits, removed = self.rank(profile, mode, snap)
        relevant = [h for h in hits if mentions_a_skill(snap.jobs[h.index], profile)]
        top = relevant[:top_k]
        details = explain(self.llm, profile, [snap.jobs[h.index] for h in top])
        results = []
        for hit, d in zip(top, details):
            results.append(
                {
                    "job": snap.jobs[hit.index].model_dump(),
                    "score": round(hit.score, 4),
                    "bm25_rank": hit.bm25_rank,
                    "dense_rank": hit.dense_rank,
                    **d,
                }
            )
        return {
            "profile": profile.model_dump(),
            "profile_source": profile_source,
            "mode": mode,
            "data": {"source": snap.source, "jobs": len(snap.jobs), **snap.info},
            "results": results,
            "weak_matches_hidden": len(hits) - len(relevant),
            "note": None if results else (
                f"No strong matches for these skills among the {len(snap.jobs)} current postings. "
                "Try different skills or relax a rule such as remote-only or a dealbreaker."),
            "filtered_out_count": len(removed),
            "filtered_out": [{"job_id": jid, "reason": why} for jid, why in removed[:MAX_FILTER_REPORT]],
        }
