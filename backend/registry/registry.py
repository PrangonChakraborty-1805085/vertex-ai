"""
VERTEX — A2A Agent Registry
Implements the A2A protocol's agent discovery mechanism.

Each agent registers itself with a card at:
  GET /.well-known/agent.json   (per-agent endpoint)
  GET /registry/agents          (orchestrator discovery)

The orchestrator calls /registry/agents on startup,
reads all capability cards, and decides which agents to invoke.
"""
from fastapi import APIRouter
from ..models import AgentCard
from ..agents.sec_agent import get_agent_card as sec_card
from ..agents.github_agent import get_agent_card as github_card
from ..agents.market_agent import get_agent_card as market_card
from ..agents.news_agent import get_agent_card as news_card
from ..agents.memory_agent import get_agent_card as memory_card   # ← added

router = APIRouter(prefix="/registry", tags=["A2A Registry"])

_AGENT_REGISTRY: dict[str, AgentCard] = {}


def _populate_registry():
    for card_fn in [sec_card, github_card, market_card, news_card, memory_card]:
        card = card_fn()
        _AGENT_REGISTRY[card.agent_id] = card


_populate_registry()


@router.get("/agents", response_model=list[AgentCard])
async def list_agents():
    """
    A2A discovery endpoint.
    Orchestrator calls this to learn what agents are available
    and what each one can do — without any hardcoded knowledge.
    """
    return list(_AGENT_REGISTRY.values())


@router.get("/agents/{agent_id}", response_model=AgentCard)
async def get_agent(agent_id: str):
    """Get a specific agent's capability card."""
    if agent_id not in _AGENT_REGISTRY:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return _AGENT_REGISTRY[agent_id]


@router.get("/health")
async def registry_health():
    return {
        "status": "ok",
        "registered_agents": len(_AGENT_REGISTRY),
        "agents": list(_AGENT_REGISTRY.keys()),
    }
