import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .llm import GroqClient
from .models import Job
from .pipeline import Matcher
from .providers import default_providers
from .retrieval import build_embedder
from .store import JobStore

ROOT = Path(__file__).resolve().parent.parent


class MatchRequest(BaseModel):
    text: str = Field(min_length=10, max_length=config.MAX_INPUT_CHARS)
    top_k: int = Field(5, ge=1, le=10)
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


def load_jobs() -> list[Job]:
    """The bundled synthetic sample jobs (used for tests, evals, and until live jobs have loaded)."""
    return [Job.model_validate(j) for j in json.loads((ROOT / "data" / "jobs.json").read_text(encoding="utf-8"))]


def create_app(matcher: Matcher | None = None, store: JobStore | None = None) -> FastAPI:
    hits: dict[str, deque] = defaultdict(deque)  # naive in-memory per-IP rate limiter

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        m, st = matcher, store
        if m is None:
            m = Matcher(load_jobs(), build_embedder(config.EMBEDDER, config.EMBED_MODEL), GroqClient())
            if config.LIVE_JOBS:
                st = JobStore(m, default_providers(), config.JOB_REFRESH_HOURS, config.MAX_LIVE_JOBS)
        app.state.matcher, app.state.store = m, st
        if st:
            st.start()
        yield
        if st:
            st.stop()

    app = FastAPI(title="Job Matcher (RAG with reasons)", lifespan=lifespan)

    @app.get("/api/health")
    def health(request: Request):
        m: Matcher = request.app.state.matcher
        st: JobStore | None = request.app.state.store
        snap = m.snapshot
        return {
            "status": "ok",
            "jobs": len(snap.jobs),
            "data_source": snap.source,
            "feeds": st.status if st else None,
            "llm_enabled": bool(m.llm and m.llm.available),
        }

    @app.post("/api/match")
    def match(req: MatchRequest, request: Request):
        # Behind a proxy (Render) the client IP is in X-Forwarded-For. This is a cost guard, not security.
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
        now = time.time()
        q = hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= config.RATE_LIMIT_PER_MIN:
            raise HTTPException(429, "Too many requests, please wait a minute.")
        q.append(now)
        return request.app.state.matcher.match(req.text, req.top_k, req.mode)

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
    return app


app = create_app()
