"""The full matching pipeline.

free text -> Profile (LLM) -> hard filters -> hybrid retrieval -> top-k -> grounded explanations
"""
from . import filters
from .explain import explain
from .llm import GroqClient
from .models import Job, Profile
from .profile import extract_profile
from .retrieval import Embedder, Hit, HybridIndex


class Matcher:
    def __init__(self, jobs: list[Job], embedder: Embedder, llm: GroqClient | None = None):
        self.jobs = jobs
        self.llm = llm
        self.index = HybridIndex(jobs, embedder)
        self.vocab = sorted({s.lower() for j in jobs for s in j.skills})

    def rank(self, profile: Profile, mode: str = "hybrid") -> tuple[list[Hit], list[tuple[str, str]]]:
        """Filters first, then retrieval over the jobs that survived. Used by the API and the evals."""
        allowed, removed = filters.apply(self.jobs, profile)
        return self.index.search(profile.query_text(), mode=mode, allowed=allowed), removed

    def match(self, text: str, top_k: int = 5, mode: str = "hybrid") -> dict:
        profile, profile_source = extract_profile(text, self.llm, self.vocab)
        hits, removed = self.rank(profile, mode)
        top = hits[:top_k]
        details = explain(self.llm, profile, [self.jobs[h.index] for h in top])
        results = []
        for hit, d in zip(top, details):
            job = self.jobs[hit.index]
            results.append(
                {
                    "job": job.model_dump(),
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
            "results": results,
            "filtered_out": [{"job_id": jid, "reason": why} for jid, why in removed],
        }
