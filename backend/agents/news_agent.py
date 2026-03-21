"""
VERTEX — News & Sentiment Agent
Fetches real headlines from NewsAPI (free dev plan: 100 req/day).

Free key: https://newsapi.org/register

Falls back to RSS feeds (no key needed) if NewsAPI is unavailable.
"""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal

import httpx
from ..config import get_llm
from langchain.schema import HumanMessage, SystemMessage

from ..config import settings
from ..models import NewsSignal, AgentCard, AgentCapability

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# Public RSS/Atom feeds as fallback (no API key needed)
RSS_FALLBACK_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
]


def get_agent_card() -> AgentCard:
    return AgentCard(
        agent_id="news-sentiment-agent",
        name="News & Sentiment Agent",
        version="1.0.0",
        description="Fetches recent news and performs LLM sentiment analysis for a company.",
        capabilities=[
            AgentCapability(
                name="fetch_news_sentiment",
                description="Fetch news and sentiment for a company",
                input_schema={"ticker": "str", "company_name": "str"},
                output_schema={"type": "NewsSignal"},
            )
        ],
        endpoint=f"http://localhost:{settings.fastapi_port}/agents/news",
        tags=["news", "sentiment", "fintech"],
    )


def _fetch_from_newsapi(ticker: str, company_name: str) -> list[dict]:
    """Fetch articles from NewsAPI."""
    if not settings.news_api_key:
        return []

    # Use both ticker and company name for better coverage
    query = f'"{ticker}" OR "{company_name}"'
    from_date = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 15,
        "apiKey": settings.news_api_key,
    }

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(NEWSAPI_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != "ok":
            logger.warning(f"NewsAPI returned non-ok status: {data.get('message', '')}")
            return []

        articles = data.get("articles", [])
        logger.info(f"[News Agent] NewsAPI returned {len(articles)} articles for {ticker}")
        return articles

    except Exception as e:
        logger.error(f"NewsAPI fetch failed: {e}")
        return []


def _fetch_yahoo_rss(ticker: str) -> list[dict]:
    """Fallback: fetch from Yahoo Finance RSS (no key required)."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    articles = []
    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "VERTEX/1.0"})
            if resp.status_code != 200:
                return []

        content = resp.text
        # Simple RSS parsing without external library
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
        for item in items[:10]:
            title = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", item)
            link = re.search(r"<link>(.*?)</link>", item)
            pub = re.search(r"<pubDate>(.*?)</pubDate>", item)
            if title:
                articles.append({
                    "title": title.group(1),
                    "url": link.group(1) if link else "",
                    "publishedAt": pub.group(1) if pub else "",
                    "source": {"name": "Yahoo Finance"},
                    "description": title.group(1),
                })

        logger.info(f"[News Agent] Yahoo RSS returned {len(articles)} items for {ticker}")
    except Exception as e:
        logger.warning(f"Yahoo RSS fallback failed: {e}")

    return articles


def _llm_analyse_sentiment(
    articles: list[dict], ticker: str, company_name: str
) -> dict:
    """Use LLM to extract sentiment and themes from article headlines."""
    if not articles:
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "key_themes": ["No recent news found"],
        }

    if settings.active_provider == "none":
        # Naive keyword-based fallback
        titles = " ".join([a.get("title", "") for a in articles]).lower()
        positive_words = ["surge", "beat", "growth", "profit", "record", "upgrade", "strong"]
        negative_words = ["drop", "miss", "loss", "cut", "downgrade", "concern", "risk", "decline"]
        pos = sum(1 for w in positive_words if w in titles)
        neg = sum(1 for w in negative_words if w in titles)
        if pos > neg * 1.5:
            sentiment = "positive"
            score = 0.4
        elif neg > pos * 1.5:
            sentiment = "negative"
            score = -0.4
        else:
            sentiment = "mixed" if pos + neg > 2 else "neutral"
            score = 0.0
        return {
            "overall_sentiment": sentiment,
            "sentiment_score": score,
            "key_themes": ["Automated keyword analysis — LLM unavailable"],
        }

    # Format headlines for LLM
    headlines_text = "\n".join([
        f"- [{a.get('source', {}).get('name', 'Unknown')}] {a.get('title', '')}"
        for a in articles[:12]
    ])

    llm = get_llm(temperature=0.1)

    prompt = f"""Analyse the sentiment of these recent news headlines about {company_name} ({ticker}).

HEADLINES:
{headlines_text}

Return a JSON object with exactly these keys:
{{
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "sentiment_score": <float from -1.0 to 1.0>,
  "key_themes": ["theme1", "theme2", "theme3"]
}}

Rules:
- sentiment_score: 1.0 = very bullish, -1.0 = very bearish, 0.0 = neutral
- key_themes: 3-5 short phrases capturing what's driving the news
- Return ONLY valid JSON, no other text."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a financial news sentiment analyser. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        import json
        text = re.sub(r"```json|```", "", response.content).strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"LLM sentiment analysis failed: {e}")
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "key_themes": ["Analysis unavailable"],
        }


async def fetch_news_sentiment(
    ticker: str,
    company_name: str,
) -> Optional[NewsSignal]:
    """
    Main entry point for the news agent.
    Tries NewsAPI first, falls back to Yahoo Finance RSS.
    """
    logger.info(f"[News Agent] Fetching news for {company_name} ({ticker})")

    # Try NewsAPI first
    articles = _fetch_from_newsapi(ticker, company_name)

    # Fallback to Yahoo RSS
    if not articles:
        logger.info(f"[News Agent] Falling back to Yahoo RSS for {ticker}")
        articles = _fetch_yahoo_rss(ticker)

    # Prepare article summaries
    article_dicts = []
    for a in articles[:10]:
        article_dict = {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", "Unknown") if isinstance(a.get("source"), dict) else str(a.get("source", "")),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
            "summary": a.get("description", ""),
        }
        if article_dict["summary"] == None or article_dict["summary"] == "":
            article_dict["summary"] = a.get("title","")
        article_dicts.append(article_dict)

    # LLM sentiment analysis
    analysis = _llm_analyse_sentiment(articles, ticker, company_name)

    # Validate sentiment type
    valid_sentiments = {"positive", "negative", "neutral", "mixed"}
    sentiment = analysis.get("overall_sentiment", "neutral")
    if sentiment not in valid_sentiments:
        sentiment = "neutral"

    logger.info(
        f"[News Agent] {ticker}: {len(articles)} articles, "
        f"sentiment={sentiment}, score={analysis.get('sentiment_score', 0)}"
    )
    return NewsSignal(
        ticker=ticker.upper(),
        company_name=company_name,
        articles=article_dicts,
        overall_sentiment=sentiment,
        sentiment_score=analysis.get("sentiment_score", 0.0),
        key_themes=analysis.get("key_themes", []),
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import json

    logging.basicConfig(level=logging.INFO)

    async def test():
        result = await fetch_news_sentiment("AAPL", "Apple Inc")
        if result:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print("No result")

    asyncio.run(test())
