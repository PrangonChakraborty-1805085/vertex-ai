"""
VERTEX — Memory Agent
Standalone A2A agent responsible for all graph memory operations.

Exposes:
  - read_context   : retrieve historical analysis context for a company
  - write_analysis : persist a completed analysis to the knowledge graph
  - get_trend      : return score trend for a company over time

This agent is discovered by the orchestrator via the A2A registry,
just like the four data agents. The orchestrator has no hardcoded
knowledge of memory operations — it delegates via agent card.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from ..config import settings
from ..models import AgentCard, AgentCapability
from ..graph.memory_store import MemoryStore
from ..graph.rag_store import (
    retrieve_context,
    index_analyst_note,
    index_past_verdict,
)

logger = logging.getLogger(__name__)


def get_agent_card() -> AgentCard:
    return AgentCard(
        agent_id="memory-agent",
        name="Memory Agent",
        version="1.0.0",
        description=(
            "Manages persistent graph memory across analysis sessions. "
            "Reads historical context before a debate and writes results "
            "after — enabling the system to reason about change over time."
        ),
        capabilities=[
            AgentCapability(
                name="read_context",
                description="Retrieve historical analysis context for a company",
                input_schema={"ticker": "str", "company_name": "str"},
                output_schema={
                    "historical_context": "str | None",
                    "prior_verdict": "str | None",
                    "analysis_count": "int",
                },
            ),
            AgentCapability(
                name="write_analysis",
                description="Persist a completed debate result to the knowledge graph",
                input_schema={
                    "ticker": "str",
                    "company_name": "str",
                    "verdict": "str",
                    "score": "float",
                    "investment_summary": "str",
                    "key_signals": "dict",
                },
                output_schema={"success": "bool", "analysis_count": "int"},
            ),
            AgentCapability(
                name="get_trend",
                description="Return score trend for a company over all past analyses",
                input_schema={"ticker": "str"},
                output_schema={"trend": "list[tuple[str, float]]"},
            ),
        ],
        endpoint=f"http://localhost:{settings.fastapi_port}/agents/memory",
        tags=["memory", "graph", "rag", "persistence"],
    )


async def read_context(ticker: str, company_name: str) -> dict:
    """
    Retrieve all historical context for a company before a debate.
    Combines NetworkX node data with ChromaDB semantic retrieval.
    Returns a structured dict the orchestrator injects into ResearchPackage.
    """
    logger.info(f"[Memory Agent] Reading context for {ticker}")

    store = MemoryStore()
    node = store.get_company_history(ticker)

    # Build graph-based context string
    graph_context_parts = []
    if node:
        count = node.get("analysis_count", 0)
        last_verdict = node.get("last_verdict")
        last_score = node.get("last_score")
        last_date = (node.get("last_analyzed") or "")[:10]

        graph_context_parts.append(
            f"This company has been analysed {count} time(s) before. "
            f"Most recent verdict: {last_verdict or 'none'} "
            f"(score: {last_score or 'n/a'}/10) on {last_date or 'unknown date'}."
        )

        # Include score trend if multiple analyses exist
        trend = store.get_score_trend(ticker)
        if len(trend) >= 2:
            trend_str = " → ".join(
                f"{score:.1f}" for _, score in reversed(trend[-5:])
            )
            graph_context_parts.append(f"Score trend (oldest→latest): {trend_str}")

    # Build semantic RAG context
    rag_context = retrieve_context(
        query=f"{company_name} investment thesis earnings growth risk valuation",
        ticker=ticker,
        collections=["sec_filings", "analyst_notes", "past_verdicts"],
        n_results=4,
    )

    # Merge
    historical_context = None
    if graph_context_parts:
        historical_context = " ".join(graph_context_parts)
        if rag_context and "No prior context" not in rag_context:
            historical_context += f"\n\nFrom knowledge base:\n{rag_context[:800]}"
    elif rag_context and "No prior context" not in rag_context:
        historical_context = f"From knowledge base:\n{rag_context[:800]}"

    logger.info(
        f"[Memory Agent] Context for {ticker}: "
        f"{'has history' if node else 'first analysis'}, "
        f"{'has RAG context' if historical_context else 'no prior context'}"
    )

    return {
        "historical_context": historical_context,
        "prior_verdict": node.get("last_verdict") if node else None,
        "analysis_count": node.get("analysis_count", 0) if node else 0,
        "has_history": node is not None,
    }


async def write_analysis(
    ticker: str,
    company_name: str,
    verdict: str,
    score: float,
    investment_summary: str,
    key_signals: Optional[dict] = None,
) -> dict:
    """
    Persist a completed analysis to both the NetworkX graph
    and the ChromaDB RAG store (for future retrieval).
    """
    logger.info(f"[Memory Agent] Writing analysis for {ticker}: {verdict} ({score}/10)")

    store = MemoryStore()

    # Write to knowledge graph
    store.update_company(
        ticker=ticker,
        company_name=company_name,
        verdict=verdict,
        score=score,
        key_signals=key_signals or {},
    )

    # Index summary into RAG so future debates can retrieve it
    if investment_summary:
        index_analyst_note(
            ticker=ticker,
            note=f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d')}] "
                 f"Verdict: {verdict} (score: {score}/10). {investment_summary}",
            source="debate_verdict",
        )
        index_past_verdict(
            ticker=ticker,
            verdict_summary=investment_summary,
            score=score,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

    node = store.get_company_history(ticker)
    count = node.get("analysis_count", 1) if node else 1

    logger.info(f"[Memory Agent] {ticker} now has {count} total analyses in graph")

    return {
        "success": True,
        "analysis_count": count,
        "ticker": ticker,
        "verdict": verdict,
        "score": score,
    }


async def get_trend(ticker: str) -> dict:
    """Return historical score trend for charting in the UI."""
    store = MemoryStore()
    trend = store.get_score_trend(ticker)
    return {
        "ticker": ticker,
        "trend": trend,
        "data_points": len(trend),
    }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import json
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("=== Memory Agent test ===\n")

        # Write two analyses
        r1 = await write_analysis(
            "AAPL", "Apple Inc", "bullish", 7.8,
            "Apple shows strong Services growth and AI integration momentum.",
            {"price": 185.0, "github_health": 8.2, "news_sentiment": 0.4},
        )
        print("Write 1:", json.dumps(r1, indent=2))

        r2 = await write_analysis(
            "AAPL", "Apple Inc", "neutral", 6.1,
            "Mixed signals — Services strong but hardware facing China headwinds.",
            {"price": 172.0, "github_health": 7.9, "news_sentiment": -0.1},
        )
        print("Write 2:", json.dumps(r2, indent=2))

        # Read back
        context = await read_context("AAPL", "Apple Inc")
        print("\nRead context:")
        print(json.dumps(context, indent=2))

        # Trend
        trend = await get_trend("AAPL")
        print("\nTrend:", json.dumps(trend, indent=2))

        # First-time company (no history)
        new_context = await read_context("NVDA", "NVIDIA Corporation")
        print("\nNew company context:")
        print(json.dumps(new_context, indent=2))

    asyncio.run(test())
