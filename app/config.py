"""Runtime configuration, read from environment variables (see .env.example)."""
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

# "fastembed" (real embeddings, ONNX, no torch) or "hashing" (dependency-free, for tests/offline)
EMBEDDER = os.getenv("EMBEDDER", "fastembed")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Live job feeds (set LIVE_JOBS=0 to use only the bundled sample jobs)
LIVE_JOBS = os.getenv("LIVE_JOBS", "1") == "1"
JOB_REFRESH_HOURS = float(os.getenv("JOB_REFRESH_HOURS", "6"))  # a few times a day keeps us within the feeds' terms
MAX_LIVE_JOBS = int(os.getenv("MAX_LIVE_JOBS", "500"))

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
MAX_INPUT_CHARS = 8000
