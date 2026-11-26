"""Runtime configuration, read from environment variables (see .env.example)."""
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

# "fastembed" (real embeddings, ONNX, no torch) or "hashing" (dependency-free, for tests/offline)
EMBEDDER = os.getenv("EMBEDDER", "fastembed")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
MAX_INPUT_CHARS = 8000
