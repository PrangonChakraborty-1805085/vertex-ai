"""
VERTEX — Graph Memory Store
Persistent knowledge graph using NetworkX.
Stores company nodes and analysis edges across sessions.

Graph structure:
  Nodes: companies (ticker as ID)
  Edges: analysis sessions (directed, from company to analysis snapshot)

Persists to disk as pickle between sessions.
"""
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import networkx as nx

from ..config import settings

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Singleton-style wrapper around a NetworkX DiGraph.
    Loads from disk on init, saves after every write.
    """

    def __init__(self):
        self.path = Path(settings.graph_persist_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph: nx.DiGraph = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> nx.DiGraph:
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    graph = pickle.load(f)
                logger.info(
                    f"[Memory] Loaded graph: {graph.number_of_nodes()} nodes, "
                    f"{graph.number_of_edges()} edges"
                )
                return graph
            except Exception as e:
                logger.warning(f"[Memory] Failed to load graph, starting fresh: {e}")
        return nx.DiGraph()

    def _save(self) -> None:
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            logger.error(f"[Memory] Save failed: {e}")

    # ── Company node management ───────────────────────────────────────────────

    def ensure_company_node(self, ticker: str, company_name: str) -> None:
        """Create company node if it doesn't exist."""
        if not self.graph.has_node(ticker):
            self.graph.add_node(
                ticker,
                company_name=company_name,
                ticker=ticker,
                first_analyzed=datetime.now(timezone.utc).isoformat(),
                last_analyzed=datetime.now(timezone.utc).isoformat(),
                analysis_count=0,
                last_verdict=None,
                last_score=None,
            )
            logger.info(f"[Memory] Created node: {ticker} ({company_name})")

    def update_company(
        self,
        ticker: str,
        company_name: str,
        verdict: str,
        score: float,
        key_signals: dict[str, Any],
    ) -> None:
        """Update company node and add a new analysis edge."""
        self.ensure_company_node(ticker, company_name)

        now = datetime.now(timezone.utc).isoformat()
        count = self.graph.nodes[ticker].get("analysis_count", 0) + 1

        # Update node attributes
        self.graph.nodes[ticker].update({
            "last_analyzed": now,
            "analysis_count": count,
            "last_verdict": verdict,
            "last_score": score,
            "company_name": company_name,
        })

        # Add analysis edge (self-loop with timestamp as key)
        edge_id = f"{ticker}_analysis_{count}"
        self.graph.add_edge(
            ticker,
            ticker,
            key=edge_id,
            analysis_date=now,
            verdict=verdict,
            score=score,
            signals=key_signals,
        )

        self._save()
        logger.info(
            f"[Memory] Updated {ticker}: verdict={verdict}, score={score}, "
            f"total_analyses={count}"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_company_history(self, ticker: str) -> Optional[dict]:
        """Return the stored metadata for a company node."""
        if not self.graph.has_node(ticker):
            return None
        return dict(self.graph.nodes[ticker])

    def get_all_analyses(self, ticker: str) -> list[dict]:
        """Return all historical analysis edges for a company."""
        if not self.graph.has_node(ticker):
            return []
        edges = []
        for u, v, data in self.graph.edges(ticker, data=True):
            if u == v:  # self-loops are our analysis records
                edges.append(data)
        return sorted(edges, key=lambda x: x.get("analysis_date", ""), reverse=True)

    def get_score_trend(self, ticker: str) -> list[tuple[str, float]]:
        """Return (date, score) pairs for charting trend over time."""
        analyses = self.get_all_analyses(ticker)
        return [
            (a.get("analysis_date", "")[:10], a.get("score", 0.0))
            for a in analyses
            if a.get("score") is not None
        ]

    def add_relationship(
        self,
        ticker_a: str,
        ticker_b: str,
        relationship: str,
        weight: float = 1.0,
    ) -> None:
        """
        Add a relationship edge between two companies.
        e.g., competitors, sector peers, supply chain.
        """
        if self.graph.has_node(ticker_a) and self.graph.has_node(ticker_b):
            self.graph.add_edge(
                ticker_a,
                ticker_b,
                relationship=relationship,
                weight=weight,
                created=datetime.now(timezone.utc).isoformat(),
            )
            self._save()

    def get_all_companies(self) -> list[dict]:
        """Return summary of all tracked companies."""
        companies = []
        for node, data in self.graph.nodes(data=True):
            if data.get("ticker"):  # skip any non-company nodes
                companies.append({
                    "ticker": node,
                    "company_name": data.get("company_name", node),
                    "last_analyzed": data.get("last_analyzed"),
                    "analysis_count": data.get("analysis_count", 0),
                    "last_verdict": data.get("last_verdict"),
                    "last_score": data.get("last_score"),
                })
        return sorted(companies, key=lambda x: x.get("last_analyzed") or "", reverse=True)

    def get_graph_stats(self) -> dict:
        """Summary stats for the UI."""
        return {
            "total_companies": self.graph.number_of_nodes(),
            "total_analyses": self.graph.number_of_edges(),
            "companies": self.get_all_companies(),
        }

    def export_for_visualization(self) -> dict:
        """
        Export graph as node/edge lists for Plotly network visualization.
        Returns {"nodes": [...], "edges": [...]}
        """
        nodes = []
        for node, data in self.graph.nodes(data=True):
            verdict = data.get("last_verdict", "unknown")
            color_map = {
                "bullish": "#1D9E75",   # teal
                "bearish": "#D85A30",   # coral
                "neutral": "#888780",   # gray
                "unknown": "#888780",
            }
            nodes.append({
                "id": node,
                "label": data.get("company_name", node),
                "score": data.get("last_score", 0),
                "verdict": verdict,
                "color": color_map.get(verdict, "#888780"),
                "size": 10 + (data.get("analysis_count", 0) * 5),
                "analyses": data.get("analysis_count", 0),
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u != v and data.get("relationship"):  # only inter-company edges
                edges.append({
                    "source": u,
                    "target": v,
                    "relationship": data.get("relationship", ""),
                    "weight": data.get("weight", 1.0),
                })

        return {"nodes": nodes, "edges": edges}


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    store = MemoryStore()

    # Simulate two analyses of Apple
    store.update_company("AAPL", "Apple Inc", "bullish", 7.8, {"price": 185.0, "github_health": 8.2})
    store.update_company("AAPL", "Apple Inc", "neutral", 6.1, {"price": 172.0, "github_health": 7.9})
    store.update_company("MSFT", "Microsoft Corp", "bullish", 8.5, {"price": 420.0})

    print("All companies:")
    print(json.dumps(store.get_all_companies(), indent=2))

    print("\nAAPL score trend:")
    print(store.get_score_trend("AAPL"))

    print("\nGraph stats:")
    print(json.dumps(store.get_graph_stats(), indent=2, default=str))
