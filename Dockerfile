FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentence-transformers langchain-groq

# Pre-download the embedding model at build time.
# Bakes the 90MB model into the image — zero cold-start delay in production.
ENV HF_HOME=/app/.cache/huggingface

COPY . .

RUN mkdir -p /app/data/chroma_db /app/data/graph_store

ENV PYTHONPATH=/app
ENV FASTAPI_PORT=8000
ENV CHROMA_PERSIST_DIR=/app/data/chroma_db
ENV GRAPH_PERSIST_PATH=/app/data/graph_store/knowledge_graph.pkl
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["sh", "-c", \
     "python -c 'from vertex.graph.rag_store import warm_up_embeddings; warm_up_embeddings()' \
     && uvicorn vertex.api.main:app --host 0.0.0.0 --port 8000"]
