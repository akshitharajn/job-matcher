FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/opt/fastembed

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so the first request is not slow.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY data ./data
COPY static ./static

# Render sets $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
