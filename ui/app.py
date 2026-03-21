"""
VERTEX — Streamlit Dashboard
Real-time multi-agent financial intelligence interface.

Run: streamlit run vertex/ui/app.py
"""
import json
import time
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VERTEX — AI Financial Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .verdict-bullish { color: #1D9E75; font-weight: 600; font-size: 1.4rem; }
  .verdict-bearish { color: #D85A30; font-weight: 600; font-size: 1.4rem; }
  .verdict-neutral  { color: #888780; font-weight: 600; font-size: 1.4rem; }
  .agent-card {
    background: var(--secondary-background-color);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
    border-left: 3px solid #1D9E75;
  }
  .agent-card.error { border-left-color: #D85A30; }
  .debate-bull {
    background: rgba(29, 158, 117, 0.08);
    border-left: 3px solid #1D9E75;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
  }
  .debate-bear {
    background: rgba(216, 90, 48, 0.08);
    border-left: 3px solid #D85A30;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
  }
  .score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, default=None):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return default


def verdict_color(verdict: str) -> str:
    return {"bullish": "#1D9E75", "bearish": "#D85A30"}.get(verdict, "#888780")


def score_to_label(score: float) -> str:
    if score >= 7.5:
        return "Strong Buy"
    if score >= 6.0:
        return "Buy"
    if score >= 4.5:
        return "Hold"
    if score >= 3.0:
        return "Sell"
    return "Strong Sell"


def _render_result(result: dict, container, ticker: str):
    """Render the full DebateResult in the Streamlit UI."""
    with container.container():
        st.markdown("---")
        st.markdown(f"### Investment verdict — {result.get('company_name', ticker)}")

        # ── Verdict header ─────────────────────────────────────────────────
        v = result.get("final_verdict", "neutral")
        score = result.get("overall_score", 5.0)
        conf = result.get("final_confidence", 0.5)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Verdict", v.upper(), delta=None)
        m2.metric("Score", f"{score:.1f} / 10")
        m3.metric("Signal", score_to_label(score))
        m4.metric("Confidence", f"{conf:.0%}")

        # ── Summary ────────────────────────────────────────────────────────
        st.markdown("#### Investment summary")
        st.write(result.get("investment_summary", ""))

        bull_col, bear_col = st.columns(2)
        with bull_col:
            st.markdown("**Bull case**")
            st.markdown(
                f'<div class="debate-bull">{result.get("bull_case", "")}</div>',
                unsafe_allow_html=True,
            )
        with bear_col:
            st.markdown("**Bear case**")
            st.markdown(
                f'<div class="debate-bear">{result.get("bear_case", "")}</div>',
                unsafe_allow_html=True,
            )

        # ── Risk factors ───────────────────────────────────────────────────
        risks = result.get("risk_factors", [])
        if risks:
            st.markdown("#### Key risk factors")
            for r in risks:
                st.markdown(f"- {r}")

        # ── Debate rounds detail ───────────────────────────────────────────
        rounds = result.get("rounds", [])
        if rounds:
            st.markdown("#### Debate transcript")
            for rd in rounds:
                round_num = rd.get("round", "?")
                verdict = rd.get("verdict", {})
                with st.expander(
                    f"Round {round_num} — Bull {verdict.get('bull_score', 0):.1f} "
                    f"vs Bear {verdict.get('bear_score', 0):.1f}"
                ):
                    bc, brc = st.columns(2)
                    with bc:
                        st.markdown("**Bull**")
                        bull_arg = rd.get("bull", {}).get("argument", "")
                        st.markdown(
                            f'<div class="debate-bull">{bull_arg}</div>',
                            unsafe_allow_html=True,
                        )
                    with brc:
                        st.markdown("**Bear**")
                        bear_arg = rd.get("bear", {}).get("argument", "")
                        st.markdown(
                            f'<div class="debate-bear">{bear_arg}</div>',
                            unsafe_allow_html=True,
                        )
                    st.caption(f"Judge: {verdict.get('reasoning', '')}")

        # ── Score chart ────────────────────────────────────────────────────
        if rounds:
            bull_scores = [rd.get("verdict", {}).get("bull_score", 0) for rd in rounds]
            bear_scores = [rd.get("verdict", {}).get("bear_score", 0) for rd in rounds]
            round_labels_chart = [f"R{rd.get('round', i+1)}" for i, rd in enumerate(rounds)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=round_labels_chart, y=bull_scores,
                name="Bull", line=dict(color="#1D9E75", width=2),
                mode="lines+markers",
            ))
            fig.add_trace(go.Scatter(
                x=round_labels_chart, y=bear_scores,
                name="Bear", line=dict(color="#D85A30", width=2),
                mode="lines+markers",
            ))
            fig.update_layout(
                title="Debate scores per round",
                yaxis=dict(range=[0, 10], title="Score"),
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ VERTEX")
    st.caption("AI Financial Intelligence Engine")
    st.divider()

    # Health check
    health = api_get("/health")
    if health:
        status = health.get("status", "unknown")
        color = "🟢" if status == "ok" else "🟡"
        st.markdown(f"{color} API **{status}**")
        if health.get("missing_keys"):
            st.warning(f"Missing keys: {', '.join(health['missing_keys'])}")
        st.caption(
            f"Graph: {health.get('graph_nodes', 0)} companies · "
            f"{health.get('graph_edges', 0)} analyses"
        )
    else:
        st.error("⚠ API offline — start with: `uvicorn vertex.api.main:app --reload`")

    st.divider()

    # Known companies shortcuts
    st.markdown("**Quick select**")
    quick = {
        "Apple": ("AAPL", "Apple Inc", "apple"),
        "Microsoft": ("MSFT", "Microsoft Corporation", "microsoft"),
        "Stripe": ("STRIPE", "Stripe Inc", "stripe"),
        "Coinbase": ("COIN", "Coinbase Global", "coinbase"),
        "Shopify": ("SHOP", "Shopify Inc", "shopify"),
    }
    for label, (ticker, name, gh_org) in quick.items():
        if st.button(label, use_container_width=True, key=f"quick_{ticker}"):
            st.session_state["input_ticker"] = ticker
            st.session_state["input_company"] = name
            st.session_state["input_github"] = gh_org

    st.divider()

    # RAG stats
    rag_stats = api_get("/rag/stats", {})
    if rag_stats:
        st.markdown("**Knowledge base**")
        for coll, count in rag_stats.items():
            st.caption(f"{coll}: {count} chunks")


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_analyse, tab_history, tab_graph = st.tabs([
    "◈ New Analysis", "📋 History", "🕸 Knowledge Graph"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: New Analysis
# ─────────────────────────────────────────────────────────────────────────────

with tab_analyse:
    st.markdown("### Analyse a company")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        ticker = st.text_input(
            "Ticker",
            value=st.session_state.get("input_ticker", "AAPL"),
            placeholder="AAPL",
        ).upper().strip()
    with col2:
        company_name = st.text_input(
            "Company name",
            value=st.session_state.get("input_company", "Apple Inc"),
            placeholder="Apple Inc",
        )
    with col3:
        github_org = st.text_input(
            "GitHub org (optional)",
            value=st.session_state.get("input_github", "apple"),
            placeholder="apple",
        )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    with col_hint:
        st.caption("Runs parallel SEC + market + GitHub + news fetch → debate loop → graph memory")

    if run_btn and ticker and company_name:
        if not health:
            st.error("API is offline. Cannot start analysis.")
        else:
            # ── Start job ──────────────────────────────────────────────────
            try:
                resp = httpx.post(
                    f"{API_BASE}/analyse",
                    json={
                        "ticker": ticker,
                        "company_name": company_name,
                        "github_org": github_org or None,
                    },
                    timeout=10,
                )
                job = resp.json()
                job_id = job["job_id"]
            except Exception as e:
                st.error(f"Failed to start analysis: {e}")
                st.stop()

            st.success(f"Job `{job_id}` started for **{company_name}** ({ticker})")

            # ── Live event stream ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Live agent feed")

            # Layout: left = event feed, right = scores
            feed_col, score_col = st.columns([3, 2])

            with feed_col:
                feed_container = st.container()
            # with score_col:
            #     score_container = st.container()
            #     score_container.markdown("**Debate scores**")
            #     score_chart = score_container.empty()
            #     round_log_container = score_container.container()

            results_area = st.empty()

            # Track state
            events_log = []
            round_bull_scores = []
            round_bear_scores = []
            round_labels = []

            # ------------- Progress Bar ---------------
            progress_text_mapping = {
                "agent_start": "Agent started",
                "debate_round": "Debate round in progress",
                "memory_update": "Updating knowledge graph",
                "complete": "Analysis complete",
                "error": "Analysis failed",
            }
            my_bar = st.progress(0, text="Starting analysis...")

            # Poll for events via status + result endpoints
            max_polls = 180
            for poll in range(max_polls):
                time.sleep(2)

                status_data = api_get(f"/status/{job_id}", {})
                events_data = api_get(f"/events/{job_id}", [])
                current_status = status_data.get("status", "queued")
                event_count = status_data.get("event_count", 0)

                # Update progress bar based on last event type
                if events_data:
                    last_event_type = events_data[-1].get("event_type", "")
                    if last_event_type == "agent_start":
                        my_bar.progress(33, text=progress_text_mapping.get("agent_start", "Agent started"))
                    elif last_event_type == "debate_round":
                        my_bar.progress(67, text=progress_text_mapping.get("debate_round", "Debate round in progress"))
                    elif last_event_type == "memory_update":
                        my_bar.progress(98, text=progress_text_mapping.get("memory_update", "Updating knowledge graph"))

                # Render feed from stored events by re-fetching
                # (Streamlit doesn't support true SSE natively — we poll)
                with feed_container:
                    feed_placeholder = st.empty()

                # Check if done
                if current_status in ("complete", "error"):
                    with feed_container:
                        if current_status == "complete":
                            st.success("✓ Analysis complete")
                        else:
                            st.error("Analysis failed")

                    # Set progress to 100% and empty the bar
                    my_bar.progress(100, text=progress_text_mapping.get(current_status, "Analysis finished"))
                    time.sleep(1)
                    my_bar.empty()

                    # Fetch and render result
                    result_data = api_get(f"/result/{job_id}")
                    if result_data:
                        _render_result(result_data, results_area, ticker)
                    break

                if poll == max_polls - 1:
                    st.warning("Analysis is taking longer than expected. Check /status endpoint.")

            # Show progress while running
            with feed_container:
                st.info(f"Status: {current_status} · Events: {event_count}")



# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: History
# ─────────────────────────────────────────────────────────────────────────────

with tab_history:
    st.markdown("### Tracked companies")

    companies = api_get("/memory/companies", [])

    if not companies:
        st.info("No companies analysed yet. Run your first analysis in the ◈ New Analysis tab.")
    else:
        # Summary table
        df = pd.DataFrame(companies)
        if not df.empty:
            # Format for display
            display_df = df[[
                "ticker", "company_name", "analysis_count",
                "last_verdict", "last_score", "last_analyzed"
            ]].copy()
            display_df.columns = ["Ticker", "Company", "Analyses", "Last Verdict", "Score", "Last Analyzed"]
            display_df["Last Analyzed"] = pd.to_datetime(
                display_df["Last Analyzed"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Per-company detail
        st.markdown("---")
        selected = st.selectbox(
            "View detailed history",
            options=[c["ticker"] for c in companies],
            format_func=lambda t: f"{t} — {next((c['company_name'] for c in companies if c['ticker'] == t), t)}",
        )

        if selected:
            history = api_get(f"/memory/history/{selected}")
            if history:
                node = history.get("company", {})
                analyses = history.get("analyses", [])

                c1, c2, c3 = st.columns(3)
                c1.metric("Total analyses", node.get("analysis_count", 0))
                c2.metric("Last verdict", node.get("last_verdict", "—"))
                c3.metric("Last score", f"{node.get('last_score', 0):.1f}/10" if node.get("last_score") else "—")

                if analyses:
                    trend_df = pd.DataFrame([
                        {
                            "Date": a.get("analysis_date", "")[:10],
                            "Score": a.get("score", 0),
                            "Verdict": a.get("verdict", ""),
                        }
                        for a in analyses
                    ])
                    fig = go.Figure(go.Scatter(
                        x=trend_df["Date"],
                        y=trend_df["Score"],
                        mode="lines+markers",
                        line=dict(color="#534AB7", width=2),
                        marker=dict(size=8),
                        hovertext=trend_df["Verdict"],
                    ))
                    fig.update_layout(
                        title=f"{selected} score trend over time",
                        yaxis=dict(range=[0, 10], title="Overall Score"),
                        height=300,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

with tab_graph:
    st.markdown("### Knowledge graph")
    st.caption("Each node is a company. Size = number of analyses. Color = last verdict.")

    graph_data = api_get("/memory/graph", {"nodes": [], "edges": []})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        st.info("Graph is empty — run some analyses to populate it.")
    else:
        # Build Plotly network graph
        # Use spring layout from networkx for positioning
        import networkx as nx
        import math

        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"])
        for e in edges:
            G.add_edge(e["source"], e["target"])

        # Compute layout
        if len(nodes) == 1:
            pos = {nodes[0]["id"]: (0.5, 0.5)}
        else:
            pos = nx.spring_layout(G, seed=42)

        # Build Plotly traces
        edge_x, edge_y = [], []
        for e in edges:
            x0, y0 = pos.get(e["source"], (0, 0))
            x1, y1 = pos.get(e["target"], (0, 0))
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="#B4B2A9"),
            hoverinfo="none",
        )

        node_x = [pos.get(n["id"], (0, 0))[0] for n in nodes]
        node_y = [pos.get(n["id"], (0, 0))[1] for n in nodes]
        node_text = [
            f"{n['label']} ({n['id']})<br>"
            f"Verdict: {n['verdict']}<br>"
            f"Score: {n['score']:.1f}/10<br>"
            f"Analyses: {n['analyses']}"
            for n in nodes
        ]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=node_text,
            text=[n["id"] for n in nodes],
            textposition="top center",
            marker=dict(
                size=[n["size"] for n in nodes],
                color=[n["color"] for n in nodes],
                line=dict(width=1.5, color="#ffffff"),
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                height=520,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown("🟢 **Bullish**")
        lc2.markdown("🔴 **Bearish**")
        lc3.markdown("⚫ **Neutral / Unknown**")

        # Node table
        st.markdown("---")
        node_df = pd.DataFrame([{
            "Ticker": n["id"],
            "Company": n["label"],
            "Verdict": n["verdict"],
            "Score": f"{n['score']:.1f}",
            "Analyses": n["analyses"],
        } for n in nodes])
        st.dataframe(node_df, hide_index=True, use_container_width=True)
