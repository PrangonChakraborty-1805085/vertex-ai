"""
VERTEX — Shared Pydantic models
All agents input/output these types. Keeps the system strongly typed.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── Agent card (A2A protocol) ─────────────────────────────────────────────────

class AgentCapability(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class AgentCard(BaseModel):
    """A2A spec: /.well-known/agent.json"""
    agent_id: str
    name: str
    version: str = "1.0.0"
    description: str
    capabilities: list[AgentCapability]
    endpoint: str
    tags: list[str] = []


# ── Data agent outputs ────────────────────────────────────────────────────────

class SECFiling(BaseModel):
    company_name: str
    ticker: str
    cik: str
    filing_type: str          # 10-K, 10-Q, 8-K, S-1
    filed_date: str
    period_of_report: str
    full_text_url: str
    summary: str              # LLM-generated summary
    risk_factors: list[str]
    key_financials: dict[str, Any]
    raw_excerpt: str          # first 3000 chars of filing


class GitHubSignal(BaseModel):
    org: str
    repo: str
    stars: int
    forks: int
    open_issues: int
    commits_last_30d: int
    contributors_count: int
    top_languages: dict[str, int]
    release_cadence_days: Optional[float]  # avg days between releases
    last_release: Optional[str]
    engineering_health_score: float  # 0-10, computed


class MarketSignal(BaseModel):
    ticker: str
    current_price: float
    price_change_pct_1d: float
    price_change_pct_30d: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    eps: Optional[float]
    earnings_surprise_pct: Optional[float]
    fifty_two_week_high: float
    fifty_two_week_low: float
    fetched_at: str


class NewsSignal(BaseModel):
    ticker: str
    company_name: str
    articles: list[dict[str, str]]   # [{title, source, url, published_at, summary}]
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    sentiment_score: float           # -1.0 to 1.0
    key_themes: list[str]


# ── Aggregated research package ───────────────────────────────────────────────

class ResearchPackage(BaseModel):
    """Everything the debate agents receive."""
    ticker: str
    company_name: str
    github_org: Optional[str]
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    sec_data: Optional[SECFiling] = None
    market_data: Optional[MarketSignal] = None
    github_data: Optional[GitHubSignal] = None
    news_data: Optional[NewsSignal] = None

    # Graph memory context (filled by memory agent)
    historical_context: Optional[str] = None
    prior_verdict: Optional[str] = None


# ── Debate models ─────────────────────────────────────────────────────────────

class DebateArgument(BaseModel):
    role: Literal["bull", "bear"]
    round_number: int
    argument: str
    supporting_evidence: list[str]
    confidence: float   # 0-1


class JudgeVerdict(BaseModel):
    round_number: int
    bull_score: float           # 0-10
    bear_score: float           # 0-10
    strongest_bull_point: str
    strongest_bear_point: str
    confidence: float           # 0-1 — if > threshold, debate ends
    reasoning: str


class DebateResult(BaseModel):
    ticker: str
    company_name: str
    rounds: list[dict[str, Any]]    # [{bull, bear, verdict}]
    final_verdict: str              # "bullish" | "bearish" | "neutral"
    final_confidence: float
    investment_summary: str         # 2-3 paragraph synthesis
    risk_factors: list[str]
    bull_case: str
    bear_case: str
    overall_score: float            # 0-10


# ── Graph memory models ───────────────────────────────────────────────────────

class CompanyNode(BaseModel):
    ticker: str
    company_name: str
    sector: Optional[str] = None
    first_analyzed: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_analyzed: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    analysis_count: int = 0


class AnalysisEdge(BaseModel):
    ticker: str
    analysis_date: str
    verdict: str
    confidence: float
    overall_score: float
    key_signals: dict[str, Any]


# ── Streaming event ───────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    event_type: Literal[
        "agent_start", "agent_complete", "agent_error",
        "debate_round", "debate_complete",
        "memory_update", "final_report"
    ]
    agent_name: str
    data: Any
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
