"""
VERTEX — Bull Analyst Agent
Builds the strongest possible investment case from the research package.
Responds to Bear's challenges each round, defending or conceding points.
"""
import logging
import re
from typing import Optional

from ..config import get_llm
from langchain.schema import HumanMessage, SystemMessage

from ..config import settings
from ..models import ResearchPackage, DebateArgument

logger = logging.getLogger(__name__)

BULL_SYSTEM_PROMPT = """You are a bullish equity analyst at a top-tier investment fund.
Your job is to build the strongest possible investment case for the given company.
You are optimistic but NOT reckless — you acknowledge risks only when cornered.
You cite specific data points from the research provided.
You argue with conviction and precision. You are never vague.
When the Bear makes a good point, you either rebut with data or concede narrowly
and pivot to a stronger bull point."""


def _build_research_summary(package: ResearchPackage) -> str:
    """Format the research package into a readable brief for the LLM."""
    parts = [f"COMPANY: {package.company_name} ({package.ticker})\n"]

    if package.market_data:
        m = package.market_data
        parts.append(
            f"MARKET DATA:\n"
            f"  Price: ${m.current_price:.2f} | 1d: {m.price_change_pct_1d:+.1f}% | "
            f"30d: {m.price_change_pct_30d:+.1f}%\n"
            f"  P/E: {m.pe_ratio or 'N/A'} | EPS: {m.eps or 'N/A'} | "
            f"EPS Surprise: {f'{m.earnings_surprise_pct:+.1f}%' if m.earnings_surprise_pct else 'N/A'}\n"
            f"  52w High: ${m.fifty_two_week_high:.2f} | 52w Low: ${m.fifty_two_week_low:.2f}\n"
        )

    if package.sec_data:
        s = package.sec_data
        parts.append(
            f"SEC FILING ({s.filing_type}, filed {s.filed_date}):\n"
            f"  Summary: {s.summary}\n"
            f"  Key financials: {s.key_financials}\n"
            f"  Risk factors: {', '.join(s.risk_factors[:3])}\n"
        )

    if package.github_data:
        g = package.github_data
        parts.append(
            f"ENGINEERING SIGNALS (GitHub: {g.org}/{g.repo}):\n"
            f"  Stars: {g.stars:,} | Forks: {g.forks:,} | "
            f"Commits (30d): {g.commits_last_30d} | Contributors: {g.contributors_count}\n"
            f"  Engineering health score: {g.engineering_health_score}/10\n"
            f"  Release cadence: {f'every {g.release_cadence_days:.0f} days' if g.release_cadence_days else 'N/A'}\n"
        )

    if package.news_data:
        n = package.news_data
        headlines = [a["title"] for a in n.articles[:5]]
        parts.append(
            f"NEWS SENTIMENT:\n"
            f"  Overall: {n.overall_sentiment} (score: {n.sentiment_score:+.2f})\n"
            f"  Key themes: {', '.join(n.key_themes)}\n"
            f"  Recent headlines: {' | '.join(headlines[:3])}\n"
        )

    if package.historical_context:
        parts.append(f"HISTORICAL CONTEXT (from memory):\n{package.historical_context}\n")

    return "\n".join(parts)


async def run_bull_argument(
    package: ResearchPackage,
    round_number: int,
    bear_argument: Optional[str] = None,
    rag_context: Optional[str] = None,
) -> DebateArgument:
    """
    Generate a bull argument for a given round.
    Round 1: opening statement
    Round 2+: response to bear's last argument
    """
    research = _build_research_summary(package)

    rag_section = f"ADDITIONAL RAG CONTEXT:\n{rag_context}" if rag_context else ""

    if round_number == 1:
        user_prompt = f"""Make your OPENING INVESTMENT CASE for {package.company_name} ({package.ticker}).

RESEARCH DATA:
{research}

{rag_section}

Structure your argument with:
1. Core thesis (1 sentence)
2. Three strongest bull points, each backed by specific data from the research
3. Your confidence level (0.0 to 1.0)

Format your response as:
THESIS: <one sentence>
POINTS:
- <point 1 with data>
- <point 2 with data>
- <point 3 with data>
CONFIDENCE: <0.0-1.0>"""
    else:
        user_prompt = f"""The Bear has argued:
"{bear_argument}"

RESEARCH DATA:
{research}

Round {round_number}: Respond to the Bear's argument.
- Rebut their strongest point with specific data
- Introduce a new bull point they haven't addressed
- If they made a valid point, concede narrowly then pivot

Format:
THESIS: <your maintained/updated thesis>
POINTS:
- <rebuttal with data>
- <new bull point>
- <any concession + pivot>
CONFIDENCE: <0.0-1.0>"""

    llm = get_llm(temperature=0.4)

    try:
        response = llm.invoke([
            SystemMessage(content=BULL_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        content = response.content

        # Extract confidence
        confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", content)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.6

        # Extract bullet points as evidence
        points = re.findall(r"^- (.+)$", content, re.MULTILINE)

        logger.info(f"[Bull Agent] Round {round_number} complete, confidence={confidence}")

        return DebateArgument(
            role="bull",
            round_number=round_number,
            argument=content,
            supporting_evidence=points[:3],
            confidence=min(max(confidence, 0.0), 1.0),
        )

    except Exception as e:
        logger.error(f"[Bull Agent] Failed: {e}")
        return DebateArgument(
            role="bull",
            round_number=round_number,
            argument=f"Bull analysis unavailable: {str(e)}",
            supporting_evidence=[],
            confidence=0.5,
        )
