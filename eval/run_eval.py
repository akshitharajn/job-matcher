"""Compare retrieval variants on a hand-labelled query set.

Usage (from the project root):
    python -m eval.run_eval --embedder hashing      # offline, no model download
    python -m eval.run_eval --embedder fastembed    # real embeddings (what production uses)

Metrics (k = 5): Recall@k, NDCG@k (graded relevance 0-2), MRR.
Labels in eval/queries.json were drafted by the author of this repo from the job texts; review and
extend them yourself - the eval is only as trustworthy as its labels.
"""
import argparse
import json
import math
from pathlib import Path

from app import config
from app.llm import GroqClient
from app.main import load_jobs
from app.models import Profile
from app.pipeline import Matcher
from app.retrieval import build_embedder

HERE = Path(__file__).parent
K = 5


def ndcg(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum(rel.get(j, 0) / math.log2(i + 2) for i, j in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def mrr(ranked: list[str], rel: dict[str, int]) -> float:
    for i, j in enumerate(ranked, start=1):
        if j in rel:
            return 1 / i
    return 0.0


def evaluate(matcher: Matcher, queries: list[dict], mode: str) -> dict[str, float]:
    r = n = m = 0.0
    for q in queries:
        profile = Profile.model_validate(q["profile"])
        hits, _ = matcher.rank(profile, mode)
        ranked = [matcher.jobs[h.index].id for h in hits]
        r += recall(ranked, q["relevant"], K)
        n += ndcg(ranked, q["relevant"], K)
        m += mrr(ranked, q["relevant"])
    c = len(queries)
    return {f"Recall@{K}": r / c, f"NDCG@{K}": n / c, "MRR": m / c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=config.EMBEDDER, choices=["fastembed", "hashing"])
    ap.add_argument("--out", default=None, help="write the markdown table to this file")
    args = ap.parse_args()

    queries = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    matcher = Matcher(load_jobs(), build_embedder(args.embedder, config.EMBED_MODEL), GroqClient(api_key=""))

    lines = [
        f"Embedder: `{args.embedder}` | queries: {len(queries)} | jobs: {len(matcher.jobs)}",
        "",
        f"| Retrieval | Recall@{K} | NDCG@{K} | MRR |",
        "|---|---|---|---|",
    ]
    for mode in ("bm25", "dense", "hybrid"):
        s = evaluate(matcher, queries, mode)
        lines.append(f"| {mode} | {s[f'Recall@{K}']:.3f} | {s[f'NDCG@{K}']:.3f} | {s['MRR']:.3f} |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
