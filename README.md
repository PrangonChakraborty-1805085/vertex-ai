# VERTEX — AI Financial Intelligence Engine

> Multi-agent financial analysis with A2A protocol, adversarial debate loops,
> graph memory. Built with LangGraph + FastAPI + Streamlit.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (port 8501)                                  │
│  Live agent feed · Debate transcript · Knowledge graph       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP + SSE
┌────────────────────────▼────────────────────────────────────┐
│  FastAPI Gateway  (port 8000)                               │
│  POST /analyse · GET /stream/{id} · GET /memory/graph       │
└────────────────────────┬────────────────────────────────────┘
                         │ A2A discovery → LangGraph
┌────────────────────────▼────────────────────────────────────┐
│  LangGraph Orchestrator                                     │
│                                                             │
│  ┌──────────────────── Phase 1 (parallel) ────────────────┐ │
│  │  SEC Agent  │  Market Agent  │  GitHub Agent  │  News  │ │
│  └─────────────────────────────────────────────────────── ┘ │
│                          ↓                                  │
│  ┌──────────────── Phase 2 (debate loop) ─────────────────┐ │
│  │    Bull ←──────────→ Bear                              │ │
│  │         ↘          ↙                                   │ │
│  │           Judge  ─────→ confidence ≥ 0.75?             │ │
│  │              ↑___________ no: loop back                │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌──────────────── Phase 3 (memory) ──────────────────────┐ │
│  │  NetworkX graph  ·  QDrant RAG  ·  Memory Agent      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Setup (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get free API keys

| Service | URL | Free Tier |
|---------|-----|-----------|
| **Groq** (required) | groq.com | Free models |
| **Alpha Vantage** (market data) | alphavantage.co/support/#api-key | 25 req/day free |
| **NewsAPI** (news) | newsapi.org/register | 100 req/day free |
| **GitHub** (engineering signals) | github.com/settings/tokens | 5000 req/hr with token |
| **SEC EDGAR** (filings) | no key needed | completely free |

### 3. Test each agent individually (recommended order)
```bash
# No keys needed for these:
python tests/test_agents.py test-sec       # Tests EDGAR filing fetch
python tests/test_agents.py test-github    # Tests GitHub API
python tests/test_agents.py test-memory    # Tests graph memory
python tests/test_agents.py test-rag       # Tests ChromaDB

# Needs Alpha Vantage key:
python tests/test_agents.py test-market

# Uses Yahoo RSS fallback if no NewsAPI key:
python tests/test_agents.py test-news

# Needs Groq key — runs full debate loop:
python tests/test_agents.py test-full
```

### 5. Run the full stack
```bash
# Terminal 1: API
uvicorn vertex.api.main:app --reload --port 8000

# Terminal 2: UI
./run.sh ui

# OR both at once:
./run.sh full
```

Open http://localhost:8501 in your browser.

---

## AI Patterns Demonstrated

### 1. Multi-agent debate loop
`Bull Agent` and `Bear Agent` receive identical data but argue opposite theses.
`Judge Agent` scores each round and triggers another round if confidence < 0.75.
LangGraph's conditional edges create the cycle — not a linear pipeline.

### 2. Simultaneous Agents coordination
All 4 data agents fire simultaneously via `asyncio.gather`.
Events stream to Streamlit via Server-Sent Events as each agent completes.

### 3. Graph-based memory across sessions
`NetworkX` DiGraph stores every company analysis as a node + edge.
Second analysis of the same company retrieves historical context and score trend.
`QDrant` RAG layer indexes filing text for semantic retrieval.

### 4. Agentic tool use with real APIs
- **SEC EDGAR**: `efts.sec.gov` + `data.sec.gov` — completely free, no key
- **Alpha Vantage**: real-time price, P/E, EPS, earnings surprise — 25 req/day free
- **GitHub API**: commit velocity, contributor growth, release cadence
- **NewsAPI + Yahoo RSS**: recent headlines, LLM sentiment scoring

### 5. A2A Protocol
Each agent exposes a `/.well-known/agent.json` card describing capabilities.
Orchestrator calls `GET /registry/agents` on startup — zero hardcoded agent knowledge.

---

## Project structure

```
vertex/
├── agents/
│   ├── sec_agent.py        SEC EDGAR fetcher + LLM parser
│   ├── github_agent.py     GitHub API signals
│   ├── market_agent.py     Alpha Vantage market data
│   ├── news_agent.py       NewsAPI + Yahoo RSS + sentiment
│   ├── bull_agent.py       Investment case builder
│   ├── bear_agent.py       Adversarial challenger
│   └── judge_agent.py      Debate scorer + synthesiser
├── graph/
│   ├── orchestrator.py     LangGraph supervisor graph
│   ├── memory_store.py     NetworkX + pickle persistence
│   └── rag_store.py        ChromaDB vector store
├── registry/
│   └── registry.py         A2A agent card server
├── api/
│   └── main.py             FastAPI + SSE endpoints
├── ui/
│   └── app.py              Streamlit dashboard
├── tests/
│   └── test_agents.py      Per-module test suite
├── config.py               Typed settings from .env
├── models.py               Shared Pydantic models
├── requirements.txt
├── .env
└── run.sh
```

---

## FAQs

**"What problem does this solve?"**
Institutional analysts spend days manually synthesising SEC filings, engineering signals, and market data into an investment view. VERTEX does this in minutes with transparent reasoning.

**"What's novel about the architecture?"**
The debate loop — Bull and Bear agents argue from the same data but opposite priors, with a Judge running multiple rounds until confidence converges. No existing financial tool does adversarial multi-agent reasoning.

**"How does A2A work here?"**
Each agent self-registers a JSON capability card. The orchestrator has zero hardcoded knowledge of agents — it discovers them at runtime, reads their schemas, and decides delegation. Adding a new agent requires only deploying it and registering its card.

**"What's the graph memory doing?"**
Every analysis updates a persistent NetworkX knowledge graph. The second time you analyze Apple, the system knows the prior verdict, score trend, and historical context — the debate agents reason about *change*, not just current state.

**"Is the data real?"**
Every data point is from live, free APIs. SEC filings are the actual EDGAR database. GitHub stats are real repository activity. Market data is from Alpha Vantage's free tier. News from NewsAPI or Yahoo Finance RSS. Nothing is simulated.
