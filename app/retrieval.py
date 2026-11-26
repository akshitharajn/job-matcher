"""Hybrid retrieval: BM25 (keywords) + dense embeddings (meaning), fused with Reciprocal Rank Fusion.

BM25 and cosine similarity are implemented in plain Python on purpose: the corpus is small, and it
keeps the algorithms readable (you should be able to explain every line in an interview).
"""
import math
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .models import Job
from .text import tokenize


# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.n, 1)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        # "+1" inside the log keeps IDF positive even for terms present in most documents.
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


# ---------------------------------------------------------------- Embedders
class Embedder(Protocol):
    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size vector. No model download; used for tests/offline.

    It has no semantic understanding (no synonyms), which is exactly why the eval shows real
    embeddings beating it on paraphrased queries.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            h = zlib.crc32(tok.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _normalize(v)

    def embed_docs(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FastEmbedEmbedder:
    """Real sentence embeddings via fastembed (ONNX runtime, small, CPU-only)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: heavy, and optional for tests

        self.model = TextEmbedding(model_name=model_name)

    def embed_docs(self, texts):
        # fastembed's default batch is 256 passages x up to 512 tokens, which can exceed a 512 MB host.
        return [_normalize([float(x) for x in v]) for v in self.model.passage_embed(texts, batch_size=8)]

    def embed_query(self, text):
        return _normalize([float(x) for x in next(iter(self.model.query_embed([text])))])


def build_embedder(name: str, model_name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    return FastEmbedEmbedder(model_name)


# ---------------------------------------------------------------- Fusion + index
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """score(d) = sum over rankers of 1 / (k + rank). Needs no score normalisation."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class Hit:
    index: int
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridIndex:
    def __init__(self, jobs: list[Job], embedder: Embedder):
        self.jobs = jobs
        self.embedder = embedder
        texts = [j.text() for j in jobs]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.vectors = embedder.embed_docs(texts)

    def search(self, query: str, mode: str = "hybrid", allowed: list[int] | None = None) -> list[Hit]:
        allowed = list(range(len(self.jobs))) if allowed is None else allowed
        if not allowed or not query.strip():
            return []

        bm = self.bm25.scores(tokenize(query))
        qv = self.embedder.embed_query(query)
        dense = {i: sum(a * b for a, b in zip(qv, self.vectors[i])) for i in allowed}

        bm_rank_list = [i for i in sorted(allowed, key=lambda i: -bm[i]) if bm[i] > 0]
        dense_rank_list = sorted(allowed, key=lambda i: -dense[i])
        bm_rank = {i: r for r, i in enumerate(bm_rank_list, start=1)}
        dense_rank = {i: r for r, i in enumerate(dense_rank_list, start=1)}

        if mode == "bm25":
            scored = {i: bm[i] for i in bm_rank_list}
        elif mode == "dense":
            scored = dense
        else:
            scored = reciprocal_rank_fusion([bm_rank_list, dense_rank_list])

        ordered = sorted(scored, key=lambda i: -scored[i])
        return [Hit(i, scored[i], bm_rank.get(i), dense_rank.get(i)) for i in ordered]
