"""
VERTEX — Module test suite
Tests each agent independently so you can validate before running the full system.

Usage:
  python -m pytest vertex/tests/test_agents.py -v
  OR run individual:
  python vertex/tests/test_agents.py sec
  python vertex/tests/test_agents.py github
  python vertex/tests/test_agents.py market
  python vertex/tests/test_agents.py news
  python vertex/tests/test_agents.py memory
  python vertex/tests/test_agents.py rag
  python vertex/tests/test_agents.py all
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg):
    print(f"{GREEN}✓ {msg}{RESET}")


def fail(msg):
    print(f"{RED}✗ {msg}{RESET}")


def warn(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def header(msg):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


# ── SEC Agent test ────────────────────────────────────────────────────────────

async def test_sec_agent():
    header("TEST: SEC Filing Agent")
    from backend.agents.sec_agent import fetch_sec_filing, _get_cik_for_ticker

    # Test CIK resolution
    cik = _get_cik_for_ticker("AAPL")
    if cik:
        ok(f"CIK resolution: AAPL → {cik}")
    else:
        fail("CIK resolution failed for AAPL")
        return

    # Test filing fetch
    print("Fetching latest 10-K for AAPL (may take ~5s)...")
    result = await fetch_sec_filing("AAPL", "10-K")
    if result:
        ok(f"Filing fetched: {result.filing_type} filed {result.filed_date}")
        ok(f"Company: {result.company_name}")
        ok(f"Summary (first 100 chars): {result.summary[:100]}...")
        if result.risk_factors:
            ok(f"Risk factors: {len(result.risk_factors)} found")
        else:
            warn("No risk factors extracted (LLM may not be configured)")
    else:
        warn("No filing returned — EDGAR may be slow, or ticker invalid")

    # Test with a different ticker
    print("Fetching for MSFT...")
    result2 = await fetch_sec_filing("MSFT", "10-Q")
    if result2:
        ok(f"MSFT filing: {result2.filing_type} filed {result2.filed_date}")
    else:
        warn("MSFT filing not returned")


# ── GitHub Agent test ─────────────────────────────────────────────────────────

async def test_github_agent():
    header("TEST: GitHub Signal Agent")
    from backend.agents.github_agent import fetch_github_signals

    # Test with well-known public repo
    print("Fetching GitHub signals for stripe/stripe-python...")
    result = await fetch_github_signals("stripe", "stripe-python")
    if result:
        ok(f"Repo: {result.org}/{result.repo}")
        ok(f"Stars: {result.stars:,} | Forks: {result.forks:,}")
        ok(f"Commits (30d): {result.commits_last_30d}")
        ok(f"Contributors: {result.contributors_count}")
        ok(f"Engineering health: {result.engineering_health_score}/10")
        if result.top_languages:
            ok(f"Languages: {list(result.top_languages.keys())[:3]}")
    else:
        fail("GitHub agent returned None — check network or rate limit")

    # Test auto-discovery
    print("Testing org auto-discovery for microsoft...")
    result2 = await fetch_github_signals("microsoft")
    if result2:
        ok(f"Auto-discovered repo: {result2.org}/{result2.repo} ({result2.stars:,} stars)")
    else:
        warn("Auto-discovery failed for microsoft")


# ── Market Agent test ─────────────────────────────────────────────────────────

async def test_market_agent():
    header("TEST: Market Signal Agent")
    from backend.config import settings
    from backend.agents.market_agent import fetch_market_signals

    if not settings.alpha_vantage_key or settings.alpha_vantage_key == "your_alpha_vantage_key_here":
        warn("ALPHA_VANTAGE_API_KEY not set — skipping market agent test")
        warn("Get free key at: https://www.alphavantage.co/support/#api-key")
        return

    print("Fetching market signals for AAPL...")
    result = await fetch_market_signals("AAPL")
    if result:
        ok(f"Ticker: {result.ticker}")
        ok(f"Price: ${result.current_price:.2f}")
        ok(f"1d change: {result.price_change_pct_1d:+.2f}%")
        ok(f"30d change: {result.price_change_pct_30d:+.2f}%")
        ok(f"P/E: {result.pe_ratio} | EPS: {result.eps}")
        ok(f"52w range: ${result.fifty_two_week_low:.2f} – ${result.fifty_two_week_high:.2f}")
        if result.earnings_surprise_pct is not None:
            ok(f"EPS surprise: {result.earnings_surprise_pct:+.1f}%")
    else:
        fail("Market agent returned None — check API key and rate limits")


# ── News Agent test ───────────────────────────────────────────────────────────

async def test_news_agent():
    header("TEST: News & Sentiment Agent")
    from backend.config import settings
    from backend.agents.news_agent import fetch_news_sentiment, _fetch_yahoo_rss

    # Test Yahoo RSS fallback (no key needed)
    print("Testing Yahoo RSS fallback for AAPL (no key needed)...")
    rss_articles = _fetch_yahoo_rss("AAPL")
    if rss_articles:
        ok(f"Yahoo RSS: {len(rss_articles)} articles")
        ok(f"Sample: {rss_articles[0].get('title', '')[:80]}")
    else:
        warn("Yahoo RSS returned nothing — may be a network issue")

    if not settings.news_api_key or settings.news_api_key == "your_newsapi_key_here":
        warn("NEWS_API_KEY not set — using RSS fallback only")

    print("Fetching full news sentiment for Apple Inc...")
    result = await fetch_news_sentiment("AAPL", "Apple Inc")
    if result:
        ok(f"Articles: {len(result.articles)}")
        ok(f"Sentiment: {result.overall_sentiment} (score: {result.sentiment_score:+.2f})")
        ok(f"Themes: {result.key_themes}")
    else:
        fail("News agent returned None")


# ── Memory Store test ─────────────────────────────────────────────────────────

def test_memory_store():
    header("TEST: Graph Memory Store")
    from backend.graph.memory_store import MemoryStore

    store = MemoryStore()
    ok(f"MemoryStore initialised, graph path: {store.path}")

    # Write some data
    store.update_company("AAPL", "Apple Inc", "bullish", 7.8, {"price": 185.0})
    store.update_company("MSFT", "Microsoft Corporation", "bullish", 8.5, {"price": 420.0})
    store.update_company("AAPL", "Apple Inc", "neutral", 6.2, {"price": 172.0})

    ok("Written 3 analysis records")

    # Read back
    history = store.get_company_history("AAPL")
    assert history is not None, "AAPL not found"
    ok(f"AAPL history: {history['analysis_count']} analyses, last verdict: {history['last_verdict']}")

    trend = store.get_score_trend("AAPL")
    ok(f"Score trend: {trend}")

    all_cos = store.get_all_companies()
    ok(f"All companies: {[c['ticker'] for c in all_cos]}")

    export = store.export_for_visualization()
    ok(f"Graph export: {len(export['nodes'])} nodes, {len(export['edges'])} edges")

    stats = store.get_graph_stats()
    ok(f"Stats: {stats['total_companies']} companies")


# ── RAG Store test ─────────────────────────────────────────────────────────────

def test_rag_store():
    header("TEST: RAG Vector Store")
    from backend.graph.rag_store import (
        index_sec_filing, index_analyst_note, retrieve_context, get_collection_stats
    )

    # Index some text
    sample_filing = """
    Apple Inc reported strong Q1 2024 results with revenue of $119.6 billion,
    up 2% year over year. iPhone revenue was $69.7 billion. Services revenue
    hit a record $23.1 billion driven by App Store and Apple Music growth.
    The company announced a $110 billion share buyback program.
    CEO Tim Cook highlighted AI features coming to iPhone in iOS 18.
    Risk factors include China revenue exposure and EU regulatory pressure.
    """ * 5  # repeat to create multiple chunks

    n = index_sec_filing(
        "AAPL",
        sample_filing,
        {"filing_date": "2024-02-01", "filing_type": "10-Q"},
    )
    ok(f"Indexed {n} chunks for AAPL")

    index_analyst_note(
        "AAPL",
        "Apple shows strong Services growth momentum. AI integration could be a catalyst.",
    )
    ok("Indexed analyst note")

    # Retrieve
    context = retrieve_context("iPhone revenue AI strategy growth", "AAPL")
    if "No prior context" not in context:
        ok(f"Retrieved context ({len(context)} chars)")
    else:
        warn("No context retrieved — ChromaDB may need local embedding model on first run")

    stats = get_collection_stats()
    ok(f"Collection stats: {stats}")


# ── Full orchestrator smoke test ──────────────────────────────────────────────

async def test_orchestrator_smoke():
    header("TEST: Orchestrator (smoke test — requires LLM key)")
    from backend.config import settings

    if not settings.groq_api_key or settings.groq_api_key == "your_openai_key_here":
        warn("OPENAI_API_KEY not set — skipping orchestrator test")
        warn("Set key in .env file to run full debate loop")
        return

    from backend.graph.orchestrator import run_analysis
    print("Running AAPL analysis (this will take 1-3 minutes)...")

    events = []
    async for event in run_analysis("AAPL", "Apple Inc", "apple"):
        events.append(event)
        print(f"  [{event.event_type}] {event.agent_name}: {str(event.data)[:80]}")

    ok(f"Analysis complete: {len(events)} events emitted")
    final = next((e for e in events if e.event_type == "final_report"), None)
    if final:
        ok(f"Final verdict: {final.data.get('final_verdict')} ({final.data.get('overall_score')}/10)")
    else:
        warn("No final_report event — check agent logs")


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_tests(target: str):
    tests = {
        "sec": test_sec_agent,
        "github": test_github_agent,
        "market": test_market_agent,
        "news": test_news_agent,
        "memory": lambda: asyncio.coroutine(lambda: test_memory_store())(),
        "rag": lambda: asyncio.coroutine(lambda: test_rag_store())(),
        "orchestrator": test_orchestrator_smoke,
    }

    if target == "all":
        test_memory_store()
        test_rag_store()
        await test_sec_agent()
        await test_github_agent()
        await test_market_agent()
        await test_news_agent()
    elif target == "memory":
        test_memory_store()
    elif target == "rag":
        test_rag_store()
    elif target in tests:
        fn = tests[target]
        if asyncio.iscoroutinefunction(fn):
            await fn()
        else:
            fn()
    else:
        print(f"Unknown test: {target}")
        print(f"Available: {list(tests.keys()) + ['all']}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    asyncio.run(run_tests(target))
