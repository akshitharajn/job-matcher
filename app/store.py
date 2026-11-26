"""Keeps the job pool fresh: fetch live feeds in the background, then swap the search index atomically.

- The app starts instantly on the bundled sample jobs, then switches to live jobs when the first fetch is done.
- Feeds are polled every JOB_REFRESH_HOURS (default 6), not per search, to respect their usage terms.
- If every feed fails, the previous pool stays in use and the error is reported in /api/health.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .models import Job
from .pipeline import Matcher
from .providers import Provider

log = logging.getLogger("jobstore")


def _round_robin(batches: list[list[Job]]) -> list[Job]:
    out = []
    for i in range(max((len(b) for b in batches), default=0)):
        out.extend(batch[i] for batch in batches if i < len(batch))
    return out


def merge_jobs(batches: list[list[Job]], cap: int) -> list[Job]:
    """Pick up to `cap` jobs: round-robin across feeds (so one feed can't fill the cap), de-duplicated by
    title + company, and jobs that mention at least one recognised skill come first. Feeds contain many
    non-technical roles; when the cap is small, filling it with those would leave nothing to match."""
    seen, unique = set(), []
    for job in _round_robin(batches):
        key = (job.title.strip().lower(), job.company.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(job)
    with_skills = [j for j in unique if j.skills]
    without = [j for j in unique if not j.skills]
    return (with_skills + without)[:cap]


class JobStore:
    def __init__(self, matcher: Matcher, providers: list[Provider], refresh_hours: float = 6.0, max_jobs: int = 500):
        self.matcher = matcher
        self.providers = providers
        self.refresh_seconds = max(refresh_hours, 0.01) * 3600
        self.max_jobs = max_jobs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status: dict = {"state": "not started", "errors": {}, "sources": {}, "refreshed_at": None}

    def refresh(self) -> dict:
        batches, sources, errors = [], {}, {}
        with ThreadPoolExecutor(max_workers=max(len(self.providers), 1)) as pool:
            futures = [(p, pool.submit(p.fetch)) for p in self.providers]
            for provider, fut in futures:
                try:
                    jobs = fut.result()
                    batches.append(jobs)
                    sources[provider.name] = len(jobs)
                except Exception as e:  # network errors, bad JSON, rate limiting...
                    errors[provider.name] = type(e).__name__
                    log.warning("feed %s failed: %s", provider.name, type(e).__name__)

        jobs = merge_jobs(batches, self.max_jobs)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if jobs:
            self.matcher.set_jobs(jobs, "live", {"sources": sources, "refreshed_at": now})
            self.status = {"state": "live", "errors": errors, "sources": sources, "refreshed_at": now, "jobs": len(jobs)}
        else:
            self.status = {**self.status, "state": "failed (keeping previous jobs)", "errors": errors}
        return self.status

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:  # never let the background thread die
                log.exception("refresh crashed")
            self._stop.wait(self.refresh_seconds)

    def start(self) -> None:
        self.status = {**self.status, "state": "loading"}
        self._thread = threading.Thread(target=self._loop, name="job-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
