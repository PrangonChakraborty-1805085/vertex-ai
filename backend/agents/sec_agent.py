"""
VERTEX — SEC Filing Agent
Fetches real filings from SEC EDGAR (no API key required).
APIs used:
  - https://efts.sec.gov/LATEST/search-index  (full-text search)
  - https://data.sec.gov/submissions/          (company metadata)
  - https://www.sec.gov/Archives/              (filing documents)
"""
import re
import logging
import httpx
from typing import Optional
from ..config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from ..models import SECFiling, AgentCard, AgentCapability

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K,10-Q,S-1,8-K"

HEADERS = {
    "User-Agent": settings.sec_user_agent,
    "Accept": "application/json",
}


def get_agent_card() -> AgentCard:
    return AgentCard(
        agent_id="sec-filing-agent",
        name="SEC Filing Agent",
        version="1.0.0",
        description="Fetches and analyses SEC EDGAR filings (10-K, 10-Q, S-1, 8-K) for public companies.",
        capabilities=[
            AgentCapability(
                name="fetch_latest_filing",
                description="Fetch and summarise the latest SEC filing for a ticker",
                input_schema={"ticker": "str", "filing_type": "str (optional)"},
                output_schema={"type": "SECFiling"},
            )
        ],
        endpoint=f"http://localhost:{settings.fastapi_port}/agents/sec",
        tags=["fintech", "legal", "sec", "edgar"],
    )


def _get_cik_for_ticker(ticker: str) -> Optional[str]:
    """
    Resolve ticker to CIK using EDGAR company search.
    EDGAR's company_tickers.json maps tickers to CIKs.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        with httpx.Client(headers=HEADERS, timeout=15) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            for _, entry in data.items():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    logger.info(f"Resolved {ticker} → CIK {cik}")
                    return cik
    except Exception as e:
        logger.error(f"CIK lookup failed for {ticker}: {e}")
    return None


def _get_latest_filing_url(cik: str, filing_type: str = "10-K") -> Optional[dict]:
    """
    Get the most recent filing of a given type for a CIK.
    Returns dict with {accession_number, filing_date, primary_document, period_of_report}
    """
    url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    try:
        with httpx.Client(headers=HEADERS, timeout=15) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        periods = filings.get("reportDate", [])
        primary_docs = filings.get("primaryDocument", [])
        company_name = data.get("name", "Unknown")

        for i, form in enumerate(forms):
            if form.upper() == filing_type.upper():
                accession = accessions[i].replace("-", "")
                return {
                    "company_name": company_name,
                    "accession": accessions[i],
                    "accession_clean": accession,
                    "filing_date": dates[i],
                    "period": periods[i] if i < len(periods) else "",
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                    "cik": cik,
                }
    except Exception as e:
        logger.error(f"Filing lookup failed for CIK {cik}: {e}")
    return None


def _fetch_filing_text(cik: str, accession_clean: str, primary_doc: str) -> str:
    """Download the actual filing document and return first ~5000 chars."""
    cik_stripped = cik.lstrip("0")
    url = f"https://www.sec.gov/Archives/edgar/full-index/{cik_stripped}/{accession_clean}/{primary_doc}"
    # Fallback URL format
    url2 = f"https://www.sec.gov/Archives/edgar/full-index/{accession_clean[:4]}/{accession_clean[4:6]}/{accession_clean[6:8]}/{accession_clean}/{primary_doc}"

    for attempt_url in [url, url2]:
        try:
            with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
                resp = client.get(attempt_url)
                if resp.status_code == 200:
                    text = resp.text
                    # Strip HTML tags for clean text
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text[:6000]
        except Exception as e:
            logger.debug(f"Filing text fetch failed at {attempt_url}: {e}")

    return ""


def _llm_analyse_filing(
    filing_text: str, company_name: str, ticker: str, filing_type: str
) -> dict:
    """Use LLM to extract structured insights from filing text."""
    if settings.active_provider == "none":
        return {
            "summary": f"[LLM unavailable] Raw filing fetched for {company_name}.",
            "risk_factors": ["API key not configured"],
            "key_financials": {},
        }

    llm = get_llm(temperature=0.1)

    prompt = f"""You are a financial analyst. Analyse this {filing_type} filing excerpt for {company_name} ({ticker}).

FILING TEXT:
{filing_text[:4000]}

Return a JSON object with exactly these keys:
{{
  "summary": "2-3 sentence executive summary of the filing",
  "risk_factors": ["top 5 risk factors as short strings"],
  "key_financials": {{
    "revenue": "if mentioned",
    "net_income": "if mentioned",
    "guidance": "if mentioned"
  }}
}}

Return ONLY the JSON, no other text."""

    try:
        response = llm.invoke([
            SystemMessage(content="You extract structured financial data from SEC filings. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        import json
        text = response.content.strip()
        # Strip markdown code fences if present
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {
            "summary": f"Filing fetched for {company_name} ({filing_type}). LLM analysis unavailable.",
            "risk_factors": [],
            "key_financials": {},
        }


async def fetch_sec_filing(
    ticker: str,
    filing_type: str = "10-K",
) -> Optional[SECFiling]:
    """
    Main entry point for the SEC agent.
    Fetches latest filing of given type for ticker.
    Falls back: 10-K → 10-Q → 8-K if preferred type not found.
    """
    logger.info(f"[SEC Agent] Fetching {filing_type} for {ticker}")

    # Step 1: resolve ticker to CIK
    cik = _get_cik_for_ticker(ticker)
    if not cik:
        logger.warning(f"Could not resolve CIK for {ticker}")
        return None

    # Step 2: get latest filing metadata with fallback chain
    filing_info = None
    for ft in [filing_type, "10-Q", "8-K"]:
        filing_info = _get_latest_filing_url(cik, ft)
        if filing_info:
            filing_type = ft
            break

    if not filing_info:
        logger.warning(f"No filings found for {ticker} (CIK {cik})")
        return None

    logger.info(
        f"[SEC Agent] Found {filing_type} filed {filing_info['filing_date']} "
        f"for {filing_info['company_name']}"
    )

    # Step 3: fetch filing text
    filing_text = _fetch_filing_text(
        cik, filing_info["accession_clean"], filing_info["primary_doc"]
    )

    # Step 4: LLM analysis
    analysis = _llm_analyse_filing(
        filing_text, filing_info["company_name"], ticker, filing_type
    )

    # Step 5: build filing URL
    accession_dashed = filing_info["accession"]
    full_text_url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar?"
        f"action=getcompany&CIK={cik}&type={filing_type}&dateb=&owner=include&count=10"
    )

    return SECFiling(
        company_name=filing_info["company_name"],
        ticker=ticker.upper(),
        cik=cik,
        filing_type=filing_type,
        filed_date=filing_info["filing_date"],
        period_of_report=filing_info["period"],
        full_text_url=full_text_url,
        summary=analysis.get("summary", ""),
        risk_factors=analysis.get("risk_factors", []),
        key_financials=analysis.get("key_financials", {}),
        raw_excerpt=filing_text[:2000],
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import json

    logging.basicConfig(level=logging.INFO)

    async def test():
        result = await fetch_sec_filing("AAPL", "10-K")
        if result:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print("No result returned")

    asyncio.run(test())
