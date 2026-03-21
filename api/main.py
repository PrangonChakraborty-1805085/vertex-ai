"""
VERTEX — FastAPI Application
Exposes:
  POST /analyse          → trigger full analysis (returns job_id)
  GET  /stream/{job_id}  → SSE stream of real-time agent events
  GET  /result/{ticker}  → latest stored result for a ticker
  GET  /memory/graph     → graph memory export for visualization
  GET  /memory/companies → all tracked companies
  GET  /registry/agents  → A2A agent discovery
  GET  /health           → system health check
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..models import StreamEvent, DebateResult
from ..graph.orchestrator import run_analysis
from ..graph.memory_store import MemoryStore
from ..graph.rag_store import get_collection_stats
from ..registry.registry import router as registry_router

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VERTEX — AI Financial Intelligence Engine",
    description="Multi-agent financial analysis with A2A protocol, debate loops, and graph memory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In prod: restrict to your Streamlit origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(registry_router)

# ── In-memory job store (per process) ────────────────────────────────────────
# In production: replace with Redis
_jobs: dict[str, dict] = {}
_job_events: dict[str, list[StreamEvent]] = {}
_job_results: dict[str, DebateResult] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    ticker: str
    company_name: str
    github_org: Optional[str] = None


class AnalysisJobResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    stream_url: str
    created_at: str


# ── Background analysis runner ────────────────────────────────────────────────

async def _run_analysis_job(job_id: str, ticker: str, company_name: str, github_org: Optional[str]):
    """Run the full VERTEX analysis and store events + final result."""
    _jobs[job_id]["status"] = "running"
    _job_events[job_id] = []

    try:
        async for event in run_analysis(ticker, company_name, github_org):
            _job_events[job_id].append(event)

            # Extract and store final result
            if event.event_type == "final_report":
                try:
                    result = DebateResult(**event.data)
                    _job_results[job_id] = result
                except Exception as e:
                    logger.error(f"Result parse error: {e}")

        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        _job_events[job_id].append(
            StreamEvent(
                event_type="agent_error",
                agent_name="Orchestrator",
                data={"error": str(e)},
            )
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/analyse", response_model=AnalysisJobResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger a new VERTEX analysis.
    Returns a job_id immediately. Stream events via GET /stream/{job_id}.
    """
    job_id = str(uuid.uuid4())[:8]
    ticker = request.ticker.upper().strip()

    _jobs[job_id] = {
        "job_id": job_id,
        "ticker": ticker,
        "company_name": request.company_name,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _job_events[job_id] = []

    background_tasks.add_task(
        _run_analysis_job,
        job_id,
        ticker,
        request.company_name,
        request.github_org,
    )

    return AnalysisJobResponse(
        job_id=job_id,
        ticker=ticker,
        status="queued",
        stream_url=f"/stream/{job_id}",
        created_at=_jobs[job_id]["created_at"],
    )


@app.get("/stream/{job_id}")
async def stream_analysis(job_id: str):
    """
    SSE endpoint — streams real-time events from the analysis job.
    Streamlit connects here and updates the UI as events arrive.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent_index = 0
        timeout_ticks = 0
        max_wait_ticks = 300  # 5 minutes at 1s polling

        while timeout_ticks < max_wait_ticks:
            events = _job_events.get(job_id, [])

            # Send any new events
            while sent_index < len(events):
                event = events[sent_index]
                yield {
                    "event": event.event_type,
                    "data": json.dumps(event.model_dump()),
                }
                sent_index += 1
                timeout_ticks = 0  # reset on activity

            # Check if job is done
            job = _jobs.get(job_id, {})
            if job.get("status") in ("complete", "error") and sent_index >= len(events):
                # Send terminal event
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "status": job["status"],
                        "job_id": job_id,
                    }),
                }
                break

            await asyncio.sleep(1)
            timeout_ticks += 1

    return EventSourceResponse(event_generator())


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status without streaming."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id].copy()
    job["event_count"] = len(_job_events.get(job_id, []))
    return job


@app.get("/events/{job_id}")
async def get_events(job_id: str):
    """
    Return all accumulated StreamEvents for a job as a JSON list.
    The UI polls this every 2s to render the live feed and debate cards.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    events = _job_events.get(job_id, [])
    return [e.model_dump() for e in events]

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get the final DebateResult for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_id not in _job_results:
        raise HTTPException(status_code=202, detail="Analysis not yet complete")
    return _job_results[job_id].model_dump()


@app.get("/memory/companies")
async def get_companies():
    """All companies in the knowledge graph."""
    store = MemoryStore()
    return store.get_all_companies()


@app.get("/memory/graph")
async def get_graph():
    """Export knowledge graph for Plotly visualization."""
    store = MemoryStore()
    return store.export_for_visualization()


@app.get("/memory/history/{ticker}")
async def get_company_history(ticker: str):
    """Full history for a specific company."""
    store = MemoryStore()
    history = store.get_company_history(ticker.upper())
    if not history:
        raise HTTPException(status_code=404, detail=f"No history for {ticker}")
    analyses = store.get_all_analyses(ticker.upper())
    return {"company": history, "analyses": analyses}


@app.get("/rag/stats")
async def get_rag_stats():
    """ChromaDB collection statistics."""
    return get_collection_stats()


@app.get("/health")
async def health():
    """System health check — validates API keys are present."""
    missing = settings.validate()
    store = MemoryStore()
    stats = store.get_graph_stats()
    return {
        "status": "ok" if not missing else "degraded",
        "missing_keys": missing,
        "graph_nodes": stats["total_companies"],
        "graph_edges": stats["total_analyses"],
        "active_jobs": len([j for j in _jobs.values() if j["status"] == "running"]),
    }


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "vertex.api.main:app",
        host="0.0.0.0",
        port=settings.fastapi_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
