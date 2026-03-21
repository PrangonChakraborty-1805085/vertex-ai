"""
VERTEX — LangGraph Orchestrator  (updated — uses memory_agent)
The core supervisor graph that coordinates all agents.

Change from previous version:
  node_memory_read and node_memory_write now delegate to memory_agent
  instead of calling memory_store directly. This completes the A2A
  pattern — the orchestrator never touches storage directly.

Graph structure:
  fetch_data (parallel) → memory_read → debate_loop (conditional cycle)
                         ↑_______________↙ (if confidence < threshold)
  → memory_write → synthesise → done
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, TypedDict, Annotated, Any, Optional
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..config import settings
from ..models import (
    ResearchPackage, DebateResult, StreamEvent,
    SECFiling, GitHubSignal, MarketSignal, NewsSignal,
    DebateArgument, JudgeVerdict,
)
from ..agents.sec_agent import fetch_sec_filing
from ..agents.github_agent import fetch_github_signals
from ..agents.market_agent import fetch_market_signals
from ..agents.news_agent import fetch_news_sentiment
from ..agents.memory_agent import read_context, write_analysis   # ← delegates here
from ..agents.bull_agent import run_bull_argument
from ..agents.bear_agent import run_bear_argument
from ..agents.judge_agent import score_debate_round, synthesise_final_verdict
from ..graph.rag_store import retrieve_context, index_sec_filing

logger = logging.getLogger(__name__)


# ── LangGraph State ───────────────────────────────────────────────────────────

class VertexState(TypedDict):
    ticker: str
    company_name: str
    github_org: Optional[str]

    sec_data: Optional[SECFiling]
    market_data: Optional[MarketSignal]
    github_data: Optional[GitHubSignal]
    news_data: Optional[NewsSignal]

    research_package: Optional[ResearchPackage]

    debate_round_no: int
    bull_arguments: Annotated[list[DebateArgument], operator.add]
    bear_arguments: Annotated[list[DebateArgument], operator.add]
    judge_verdicts: Annotated[list[JudgeVerdict], operator.add]
    debate_rounds_log: Annotated[list[dict], operator.add]

    historical_context: Optional[str]
    prior_verdict: Optional[str]

    debate_result: Optional[DebateResult]
    events: Annotated[list[StreamEvent], operator.add]
    error: Optional[str]


# ── Data fetch nodes ──────────────────────────────────────────────────────────

async def _fetch_sec(state: VertexState) -> dict:
    ticker = state["ticker"]
    try:
        data = await fetch_sec_filing(ticker)
        if data:
            index_sec_filing(
                ticker, data.raw_excerpt,
                {"filing_date": data.filed_date, "filing_type": data.filing_type},
            )
        event = StreamEvent(
            event_type="agent_complete" if data else "agent_error",
            agent_name="SEC Filing Agent",
            data={"status": "ok" if data else "no_data",
                  "filing_type": data.filing_type if data else None},
        )
        return {"sec_data": data, "events": [event]}
    except Exception as e:
        event = StreamEvent(event_type="agent_error", agent_name="SEC Filing Agent",
                            data={"error": str(e)})
        return {"sec_data": None, "events": [event]}


async def _fetch_market(state: VertexState) -> dict:
    try:
        data = await fetch_market_signals(state["ticker"])
        event = StreamEvent(
            event_type="agent_complete" if data else "agent_error",
            agent_name="Market Signal Agent",
            data={"status": "ok" if data else "no_key",
                  "price": data.current_price if data else None},
        )
        return {"market_data": data, "events": [event]}
    except Exception as e:
        event = StreamEvent(event_type="agent_error", agent_name="Market Signal Agent",
                            data={"error": str(e)})
        return {"market_data": None, "events": [event]}


async def _fetch_github(state: VertexState) -> dict:
    org = state.get("github_org") or state["ticker"].lower()
    try:
        data = await fetch_github_signals(org)
        event = StreamEvent(
            event_type="agent_complete" if data else "agent_error",
            agent_name="GitHub Signal Agent",
            data={"status": "ok" if data else "not_found",
                  "health": data.engineering_health_score if data else None},
        )
        return {"github_data": data, "events": [event]}
    except Exception as e:
        event = StreamEvent(event_type="agent_error", agent_name="GitHub Signal Agent",
                            data={"error": str(e)})
        return {"github_data": None, "events": [event]}


async def _fetch_news(state: VertexState) -> dict:
    try:
        data = await fetch_news_sentiment(state["ticker"], state["company_name"])
        event = StreamEvent(
            event_type="agent_complete" if data else "agent_error",
            agent_name="News Sentiment Agent",
            data={"status": "ok" if data else "no_key",
                  "sentiment": data.overall_sentiment if data else None},
        )
        return {"news_data": data, "events": [event]}
    except Exception as e:
        event = StreamEvent(event_type="agent_error", agent_name="News Sentiment Agent",
                            data={"error": str(e)})
        return {"news_data": None, "events": [event]}


async def node_parallel_fetch(state: VertexState) -> dict:
    """Fan out all four data agents simultaneously."""
    logger.info("[Orchestrator] Starting parallel data fetch")

    results = await asyncio.gather(
        _fetch_sec(state),
        _fetch_market(state),
        _fetch_github(state),
        _fetch_news(state),
        return_exceptions=True,
    )

    merged = {
        "sec_data": None, "market_data": None,
        "github_data": None, "news_data": None,
        "events": [],
    }
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Parallel fetch exception: {result}")
            continue
        if isinstance(result, dict):
            for k, v in result.items():
                if k == "events":
                    merged["events"].extend(v or [])
                else:
                    merged[k] = v
    return merged


# ── Memory nodes — now delegate to memory_agent ───────────────────────────────

async def node_memory_read(state: VertexState) -> dict:
    """
    Delegate to memory_agent.read_context.
    Orchestrator discovers this capability from the agent's card —
    it does not call storage directly.
    """
    ticker = state["ticker"]
    logger.info(f"[Orchestrator] Delegating memory read to Memory Agent for {ticker}")

    # A2A call — orchestrator asks memory agent for context
    memory_result = await read_context(ticker, state["company_name"])

    historical_context = memory_result.get("historical_context")
    prior_verdict = memory_result.get("prior_verdict")

    # Assemble research package
    package = ResearchPackage(
        ticker=ticker,
        company_name=state["company_name"],
        github_org=state.get("github_org"),
        sec_data=state.get("sec_data"),
        market_data=state.get("market_data"),
        github_data=state.get("github_data"),
        news_data=state.get("news_data"),
        historical_context=historical_context,
        prior_verdict=prior_verdict,
    )

    event = StreamEvent(
        event_type="memory_update",
        agent_name="Memory Agent",
        data={
            "action": "read",
            "has_history": memory_result.get("has_history", False),
            "analysis_count": memory_result.get("analysis_count", 0),
            "prior_verdict": prior_verdict,
        },
    )

    return {
        "research_package": package,
        "historical_context": historical_context,
        "prior_verdict": prior_verdict,
        "events": [event],
    }


async def node_debate_round(state: VertexState) -> dict:
    """Execute one round of Bull vs Bear debate."""
    round_num = state.get("debate_round_no", 0) + 1
    package = state["research_package"]

    prev_bull = state["bull_arguments"][-1].argument if state.get("bull_arguments") else None
    prev_bear = state["bear_arguments"][-1].argument if state.get("bear_arguments") else None

    rag_context = retrieve_context(
        f"{package.company_name} investment risk valuation", package.ticker
    )

    event_start = StreamEvent(
        event_type="debate_round",
        agent_name="Debate Chamber",
        data={"round": round_num, "status": "starting"},
    )

    logger.info(f"[Orchestrator] Debate round {round_num}")

    bull_result, bear_result = await asyncio.gather(
        run_bull_argument(package, round_num, prev_bear, rag_context),
        run_bear_argument(package, round_num, prev_bull, rag_context),
    )

    verdict = await score_debate_round(round_num, bull_result, bear_result)

    round_log = {
        "round": round_num,
        "bull": bull_result.model_dump(),
        "bear": bear_result.model_dump(),
        "verdict": verdict.model_dump(),
    }

    event_complete = StreamEvent(
        event_type="debate_round",
        agent_name="Debate Chamber",
        data={
            "round": round_num,
            "status": "complete",
            "bull_score": verdict.bull_score,
            "bear_score": verdict.bear_score,
            "confidence": verdict.confidence,
        },
    )

    return {
        "debate_round_no": round_num,
        "bull_arguments": [bull_result],
        "bear_arguments": [bear_result],
        "judge_verdicts": [verdict],
        "debate_rounds_log": [round_log],
        "events": [event_start, event_complete],
    }


async def node_synthesise(state: VertexState) -> dict:
    """Produce the final investment report."""
    logger.info("[Orchestrator] Synthesising final verdict")

    result = await synthesise_final_verdict(
        ticker=state["ticker"],
        company_name=state["company_name"],
        rounds=state.get("debate_rounds_log", []),
        all_bull_args=state.get("bull_arguments", []),
        all_bear_args=state.get("bear_arguments", []),
        all_verdicts=state.get("judge_verdicts", []),
    )

    event = StreamEvent(
        event_type="final_report",
        agent_name="Orchestrator",
        data=result.model_dump(),
    )
    return {"debate_result": result, "events": [event]}


async def node_memory_write(state: VertexState) -> dict:
    """
    Delegate to memory_agent.write_analysis.
    Orchestrator never touches storage directly.
    """
    result = state.get("debate_result")
    if not result:
        return {"events": []}

    ticker = state["ticker"]
    logger.info(f"[Orchestrator] Delegating memory write to Memory Agent for {ticker}")

    # A2A call — orchestrator asks memory agent to persist
    write_result = await write_analysis(
        ticker=ticker,
        company_name=state["company_name"],
        verdict=result.final_verdict,
        score=result.overall_score,
        investment_summary=result.investment_summary,
        key_signals={
            "market_price": state["market_data"].current_price if state.get("market_data") else None,
            "github_health": state["github_data"].engineering_health_score if state.get("github_data") else None,
            "news_sentiment": state["news_data"].sentiment_score if state.get("news_data") else None,
        },
    )

    event = StreamEvent(
        event_type="memory_update",
        agent_name="Memory Agent",
        data={
            "action": "write",
            "ticker": ticker,
            "verdict": result.final_verdict,
            "analysis_count": write_result.get("analysis_count", 1),
        },
    )
    return {"events": [event]}


# ── Conditional edge ──────────────────────────────────────────────────────────

def should_continue_debate(state: VertexState) -> str:
    verdicts = state.get("judge_verdicts", [])
    round_num = state.get("debate_round_no", 0)

    if round_num >= settings.debate_max_rounds:
        logger.info(f"[Orchestrator] Max rounds reached → synthesise")
        return "synthesise"

    if verdicts and verdicts[-1].confidence >= settings.debate_confidence_threshold:
        logger.info(
            f"[Orchestrator] Confidence {verdicts[-1].confidence:.2f} "
            f">= {settings.debate_confidence_threshold} → synthesise"
        )
        return "synthesise"

    logger.info(f"[Orchestrator] Round {round_num} → continue debate")
    return "continue_debate"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_vertex_graph() -> StateGraph:
    graph = StateGraph(VertexState)

    graph.add_node("parallel_fetch", node_parallel_fetch)
    graph.add_node("memory_read", node_memory_read)
    graph.add_node("debate_round", node_debate_round)
    graph.add_node("synthesise", node_synthesise)
    graph.add_node("memory_write", node_memory_write)

    graph.set_entry_point("parallel_fetch")
    graph.add_edge("parallel_fetch", "memory_read")
    graph.add_edge("memory_read", "debate_round")
    graph.add_conditional_edges(
        "debate_round",
        should_continue_debate,
        {
            "continue_debate": "debate_round",
            "synthesise": "synthesise",
        },
    )
    graph.add_edge("synthesise", "memory_write")
    graph.add_edge("memory_write", END)

    return graph.compile(checkpointer=MemorySaver())


# ── Streaming run ─────────────────────────────────────────────────────────────

async def run_analysis(
    ticker: str,
    company_name: str,
    github_org: Optional[str] = None,
) -> AsyncGenerator[StreamEvent, None]:
    graph = build_vertex_graph()

    initial_state = VertexState(
        ticker=ticker.upper(),
        company_name=company_name,
        github_org=github_org,
        sec_data=None, market_data=None,
        github_data=None, news_data=None,
        research_package=None,
        debate_round_no=0,
        bull_arguments=[], bear_arguments=[],
        judge_verdicts=[], debate_rounds_log=[],
        historical_context=None, prior_verdict=None,
        debate_result=None, events=[], error=None,
    )

    config = {
        "configurable": {
            "thread_id": f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    }

    yield StreamEvent(
        event_type="agent_start",
        agent_name="Orchestrator",
        data={"ticker": ticker, "company": company_name, "message": "Analysis started"},
    )

    seen_events: set[str] = set()
    async for chunk in graph.astream(initial_state, config=config):
        for _, node_output in chunk.items():
            if isinstance(node_output, dict):
                for event in node_output.get("events", []):
                    key = f"{event.event_type}_{event.agent_name}_{event.timestamp}"
                    if key not in seen_events:
                        seen_events.add(key)
                        yield event
