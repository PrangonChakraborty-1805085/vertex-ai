"""
VERTEX — rag_store.py (Qdrant version)
Replaces ChromaDB with Qdrant — works on Windows, Linux, Docker, cloud.

Local dev:  in-memory or file-based Qdrant (no server needed)
Production: Qdrant Docker or Qdrant Cloud free tier

Install: pip install qdrant-client cohere

Qdrant modes (set QDRANT_MODE in .env):
  "memory"  → in-memory, lost on restart, good for testing
  "local"   → file-based, persists to disk, good for dev (default)
  "server"  → connects to Qdrant server/Docker/Cloud
"""
import logging
import os
import uuid
import traceback
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
COHERE_API_KEY  = os.getenv("COHERE_API_KEY", "")
EMBEDDING_MODEL = getattr(settings, "embedding_model", "embed-english-light-v3.0")
QDRANT_MODE     = os.getenv("QDRANT_MODE", "local")           # memory | local | server
QDRANT_PATH     = os.getenv("QDRANT_PATH", "./data/qdrant_db") # for local mode
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")  # for server mode
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")             # for Qdrant Cloud
EMBEDDING_DIM   = 384   # embed-english-light-v3.0 dimension

COLLECTIONS = ["sec_filings", "analyst_notes", "past_verdicts"]

# ── Cohere client (singleton) ─────────────────────────────────────────────────
_cohere_client = None

def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(COHERE_API_KEY)
        logger.info(f"[RAG] Cohere client ready (SDK v{cohere.__version__})")
    return _cohere_client


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using Cohere, returning plain list[list[float]].
    Handles both Cohere SDK v4 and v5 response shapes.
    """
    co = _get_cohere()
    logger.info(f"[RAG] Embedding {len(texts)} texts...")

    resp = co.embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        input_type="search_document",
        embedding_types=["float"],
    )

    # Cohere v5 returns resp.embeddings.float_
    # Cohere v4 returns resp.embeddings as a list directly
    raw = None
    emb = resp.embeddings
    if hasattr(emb, "float_") and emb.float_ is not None:
        raw = emb.float_
    elif isinstance(emb, list):
        raw = emb
    else:
        raw = list(emb)

    result = []
    for vec in raw:
        if hasattr(vec, "tolist"):
            result.append(vec.tolist())
        else:
            result.append([float(x) for x in vec])

    logger.info(f"[RAG] Got {len(result)} embeddings dim={len(result[0]) if result else 0}")
    return result


def _embed_query(query: str) -> list[float]:
    """Embed a single query string (uses search_query input_type)."""
    co = _get_cohere()
    resp = co.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    emb = resp.embeddings
    if hasattr(emb, "float_") and emb.float_ is not None:
        raw = emb.float_[0]
    elif isinstance(emb, list):
        raw = emb[0]
    else:
        raw = list(emb)[0]

    if hasattr(raw, "tolist"):
        return raw.tolist()
    return [float(x) for x in raw]


# ── Qdrant client (singleton) ─────────────────────────────────────────────────
_qdrant_client = None

def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    mode = QDRANT_MODE.lower()

    if mode == "memory":
        client = QdrantClient(":memory:")
        logger.info("[RAG] Qdrant: in-memory mode")

    elif mode == "local":
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=QDRANT_PATH)
        logger.info(f"[RAG] Qdrant: local file mode at {QDRANT_PATH}")

    elif mode == "server":
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        client = QdrantClient(**kwargs)
        logger.info(f"[RAG] Qdrant: server mode at {QDRANT_URL}")

    else:
        raise ValueError(f"Unknown QDRANT_MODE: {mode}. Use: memory | local | server")

    # Ensure all collections exist
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"[RAG] Created collection: {name}")

    _qdrant_client = client
    return _qdrant_client


def reset_client_cache() -> None:
    """Force re-init — useful in tests."""
    global _qdrant_client, _cohere_client
    _qdrant_client = None
    _cohere_client = None


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if not text:
        return []
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ── Index functions ───────────────────────────────────────────────────────────

def index_sec_filing(ticker: str, filing_text: str, metadata: dict) -> int:
    """Index SEC filing text. Returns number of chunks indexed."""
    if not filing_text:
        return 0
    try:
        from qdrant_client.models import PointStruct

        logger.info(f"[RAG] index_sec_filing: {ticker}")
        client = _get_qdrant()
        chunks = _chunk_text(filing_text)
        if not chunks:
            return 0

        logger.info(f"[RAG] {len(chunks)} chunks to embed and index")

        # Embed in batches of 48 (Cohere's recommended batch size)
        EMBED_BATCH = 48
        all_embeddings = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start: start + EMBED_BATCH]
            embeddings = _embed(batch)
            all_embeddings.extend(embeddings)

        # Build Qdrant points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ticker}_{metadata.get('filing_date','x')}_{i}")),
                vector=embedding,
                payload={
                    **metadata,
                    "ticker": ticker,
                    "chunk_index": i,
                    "text": chunk,
                },
            ))

        # Upsert to Qdrant in batches of 100
        UPSERT_BATCH = 100
        for start in range(0, len(points), UPSERT_BATCH):
            batch_points = points[start: start + UPSERT_BATCH]
            client.upsert(collection_name="sec_filings", points=batch_points)
            logger.info(f"[RAG] Upserted {start + len(batch_points)}/{len(points)} points")

        logger.info(f"[RAG] index_sec_filing done: {len(points)} chunks ✓")
        return len(points)

    except Exception as e:
        logger.error(f"[RAG] index_sec_filing failed: {e}\n{traceback.format_exc()}")
        return 0


def index_analyst_note(ticker: str, note: str, source: str = "llm_analysis") -> None:
    """Index an LLM-generated analyst note."""
    if not note:
        return
    try:
        from datetime import datetime, timezone
        from qdrant_client.models import PointStruct

        client = _get_qdrant()
        embedding = _embed([note])[0]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ticker}_{source}_{timestamp}"))

        client.upsert(
            collection_name="analyst_notes",
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"ticker": ticker, "source": source, "text": note},
            )],
        )
        logger.info(f"[RAG] index_analyst_note: {ticker} ✓")
    except Exception as e:
        logger.error(f"[RAG] index_analyst_note failed: {e}\n{traceback.format_exc()}")


def index_past_verdict(ticker: str, verdict_summary: str, score: float, date: str) -> None:
    """Index a past debate verdict for the graph memory feedback loop."""
    if not verdict_summary:
        return
    try:
        from qdrant_client.models import PointStruct

        client = _get_qdrant()
        embedding = _embed([verdict_summary])[0]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ticker}_verdict_{date}"))

        client.upsert(
            collection_name="past_verdicts",
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"ticker": ticker, "score": score, "date": date, "text": verdict_summary},
            )],
        )
        logger.info(f"[RAG] index_past_verdict: {ticker} ✓")
    except Exception as e:
        logger.error(f"[RAG] index_past_verdict failed: {e}\n{traceback.format_exc()}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    ticker: str,
    collections: list[str] = None,
    n_results: int = 5,
) -> str:
    """Retrieve relevant context for RAG injection into debate agents."""
    if collections is None:
        collections = COLLECTIONS
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = _get_qdrant()
        query_embedding = _embed_query(query)
        results_parts = []

        for coll_name in collections:
            try:
                # Check collection has documents
                info = client.get_collection(coll_name)
                if info.points_count == 0:
                    continue

                # Filter by ticker
                ticker_filter = Filter(
                    must=[FieldCondition(
                        key="ticker",
                        match=MatchValue(value=ticker),
                    )]
                )

                hits = client.search(
                    collection_name=coll_name,
                    query_vector=query_embedding,
                    query_filter=ticker_filter,
                    limit=min(n_results, 3),
                    score_threshold=0.2,   # cosine similarity threshold (0-1, higher = more similar)
                )

                for hit in hits:
                    text = hit.payload.get("text", "")
                    if text:
                        results_parts.append(
                            f"[{coll_name}] (relevance: {round(hit.score, 2)})\n{text}"
                        )

            except Exception as e:
                logger.warning(f"[RAG] Query failed for {coll_name}: {e}")

    except Exception as e:
        logger.error(f"[RAG] retrieve_context error: {e}")
        return f"No prior context found for {ticker} in knowledge base."

    if not results_parts:
        return f"No prior context found for {ticker} in knowledge base."
    return "\n\n---\n\n".join(results_parts)


def get_collection_stats() -> dict:
    """Return document counts per collection."""
    try:
        client = _get_qdrant()
        return {
            c.name: client.get_collection(c.name).points_count
            for c in client.get_collections().collections
        }
    except Exception as e:
        logger.error(f"[RAG] Stats failed: {e}")
        return {}


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print(f"\nQdrant mode    : {QDRANT_MODE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Cohere key     : {'SET' if COHERE_API_KEY else 'NOT SET'}\n")

    sample = (
        "Apple Inc reported record revenue of $119.6 billion for Q1 2024. "
        "Services revenue hit $23.1 billion. CEO Tim Cook highlighted AI "
        "features in iOS 18. Risk: China exposure and EU regulatory pressure. "
    ) * 5

    print("1. index_sec_filing...")
    n = index_sec_filing("AAPL", sample, {"filing_date": "2024-01-30", "filing_type": "10-Q"})
    print(f"   {'✓' if n > 0 else '✗'} {n} chunks\n")

    print("2. index_analyst_note...")
    index_analyst_note("AAPL", "Strong Services growth. AI integration is a key catalyst.")
    print()

    print("3. retrieve_context...")
    ctx = retrieve_context("iPhone revenue AI strategy", "AAPL")
    if "No prior context" not in ctx:
        print(f"   ✓ Retrieved {len(ctx)} chars")
        print(f"   Preview: {ctx[:300]}...")
    else:
        print(f"   – {ctx}")

    print("\n4. Collection stats:")
    for name, count in get_collection_stats().items():
        print(f"   {name}: {count} points")
