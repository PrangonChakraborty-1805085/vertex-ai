"""
VERTEX — Market Signal Agent
Fetches real stock data from Alpha Vantage (free tier: 25 req/day).

Free API key: https://www.alphavantage.co/support/#api-key

Endpoints used (all free tier):
  - GLOBAL_QUOTE          → current price, volume, change
  - OVERVIEW              → market cap, P/E, EPS, 52-week range
  - EARNINGS              → EPS surprise
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from ..config import settings
from ..models import MarketSignal, AgentCard, AgentCapability

logger = logging.getLogger(__name__)

AV_BASE = "https://www.alphavantage.co/query"


def get_agent_card() -> AgentCard:
    return AgentCard(
        agent_id="market-signal-agent",
        name="Market Signal Agent",
        version="1.0.0",
        description="Fetches real-time and historical market data: price, volume, P/E, EPS surprise.",
        capabilities=[
            AgentCapability(
                name="fetch_market_signals",
                description="Fetch current market signals for a ticker",
                input_schema={"ticker": "str"},
                output_schema={"type": "MarketSignal"},
            )
        ],
        endpoint=f"http://localhost:{settings.fastapi_port}/agents/market",
        tags=["fintech", "market", "stocks"],
    )


def _av_get(function: str, ticker: str, extra: dict = None) -> Optional[dict]:
    """Make a single Alpha Vantage API call."""
    if not settings.alpha_vantage_key:
        logger.warning("Alpha Vantage key not configured — market data unavailable")
        return None

    params = {
        "function": function,
        "symbol": ticker,
        "apikey": settings.alpha_vantage_key,
    }
    if extra:
        params.update(extra)

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(AV_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        # Alpha Vantage puts error info in a "Note" or "Information" field
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit hit: {data['Note'][:80]}")
            return None
        if "Information" in data:
            logger.warning(f"Alpha Vantage info: {data['Information'][:80]}")
            return None

        return data
    except Exception as e:
        logger.error(f"Alpha Vantage call failed ({function}): {e}")
        return None


def _safe_float(val, default=None) -> Optional[float]:
    """Parse a value that might be 'None', '-', or a number string."""
    if val is None or val in ("None", "-", "N/A", ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


async def fetch_market_signals(ticker: str) -> Optional[MarketSignal]:
    """
    Main entry point for the market agent.
    Makes up to 3 Alpha Vantage calls (uses daily quota carefully).
    """
    logger.info(f"[Market Agent] Fetching signals for {ticker}")

    if not settings.alpha_vantage_key:
        logger.warning(f"[Market Agent] No API key — returning None for {ticker}")
        return None

    # Call 1: Global Quote (current price, change, volume)
    quote_data = _av_get("GLOBAL_QUOTE", ticker)
    if not quote_data:
        return None

    quote = quote_data.get("Global Quote", {})
    if not quote or not quote.get("05. price"):
        logger.warning(f"[Market Agent] Empty quote for {ticker} — may be delisted or typo")
        return None

    current_price = _safe_float(quote.get("05. price"), 0.0)
    price_change_pct = _safe_float(quote.get("10. change percent", "0%").replace("%", ""), 0.0)
    volume = int(_safe_float(quote.get("06. volume"), 0) or 0)
    prev_close = _safe_float(quote.get("08. previous close"), current_price)

    # Call 2: Company Overview (market cap, P/E, 52-week range, EPS)
    overview_data = _av_get("OVERVIEW", ticker)
    overview = overview_data if overview_data else {}

    market_cap = _safe_float(overview.get("MarketCapitalization"))
    pe_ratio = _safe_float(overview.get("PERatio"))
    eps = _safe_float(overview.get("EPS"))
    week_high = _safe_float(overview.get("52WeekHigh"), current_price * 1.1)
    week_low = _safe_float(overview.get("52WeekLow"), current_price * 0.9)

    # Call 3: Earnings (EPS surprise — last quarter)
    earnings_data = _av_get("EARNINGS", ticker)
    earnings_surprise_pct = None
    if earnings_data and "quarterlyEarnings" in earnings_data:
        quarterly = earnings_data["quarterlyEarnings"]
        if quarterly:
            latest = quarterly[0]
            reported = _safe_float(latest.get("reportedEPS"))
            estimated = _safe_float(latest.get("estimatedEPS"))
            if reported is not None and estimated and estimated != 0:
                earnings_surprise_pct = round((reported - estimated) / abs(estimated) * 100, 2)

    # Compute 30d price change using daily time series (1 call, stays in free tier)
    price_change_30d = 0.0
    ts_data = _av_get("TIME_SERIES_DAILY", ticker, {"outputsize": "compact"})
    if ts_data and "Time Series (Daily)" in ts_data:
        ts = ts_data["Time Series (Daily)"]
        sorted_dates = sorted(ts.keys(), reverse=True)
        if len(sorted_dates) >= 22:
            price_30d_ago = _safe_float(ts[sorted_dates[21]].get("4. close"), current_price)
            if price_30d_ago and price_30d_ago != 0:
                price_change_30d = round((current_price - price_30d_ago) / price_30d_ago * 100, 2)

    logger.info(
        f"[Market Agent] {ticker}: price={current_price}, 1d_chg={price_change_pct}%, "
        f"30d_chg={price_change_30d}%, pe={pe_ratio}"
    )

    return MarketSignal(
        ticker=ticker.upper(),
        current_price=current_price,
        price_change_pct_1d=price_change_pct,
        price_change_pct_30d=price_change_30d,
        volume=volume,
        market_cap=market_cap,
        pe_ratio=pe_ratio,
        eps=eps,
        earnings_surprise_pct=earnings_surprise_pct,
        fifty_two_week_high=week_high or current_price,
        fifty_two_week_low=week_low or current_price,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import json

    logging.basicConfig(level=logging.INFO)

    async def test():
        result = await fetch_market_signals("AAPL")
        if result:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print("No result — check ALPHA_VANTAGE_API_KEY in .env")

    asyncio.run(test())
