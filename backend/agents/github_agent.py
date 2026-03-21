"""
VERTEX — GitHub Signal Agent
Fetches engineering health signals from GitHub's public API.

Free tier:
  - Unauthenticated: 60 req/hr
  - With GITHUB_TOKEN: 5000 req/hr (recommended)

No GitHub token? We gracefully degrade — still fetches stars, forks,
open issues, language breakdown from unauthenticated endpoints.
"""
import logging
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..config import settings
from ..models import GitHubSignal, AgentCard, AgentCapability

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


def get_agent_card() -> AgentCard:
    return AgentCard(
        agent_id="github-signal-agent",
        name="GitHub Signal Agent",
        version="1.0.0",
        description="Fetches engineering health metrics from GitHub: commit velocity, contributor growth, release cadence.",
        capabilities=[
            AgentCapability(
                name="fetch_github_signals",
                description="Fetch engineering signals for a GitHub org/repo",
                input_schema={"org": "str", "repo": "str (optional)"},
                output_schema={"type": "GitHubSignal"},
            )
        ],
        endpoint=f"http://localhost:{settings.fastapi_port}/agents/github",
        tags=["engineering", "github", "software"],
    )


def _make_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _find_main_repo(client: httpx.Client, org: str) -> Optional[dict]:
    """
    Find the most significant public repo for an org.
    Strategy: most-starred repo that isn't just a demo/docs.
    """
    url = f"{GITHUB_API}/orgs/{org}/repos"
    params = {"type": "public", "sort": "stargazers", "per_page": 10}
    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        repos = resp.json()
        if not repos:
            # Try as a user (some "orgs" are user accounts)
            url = f"{GITHUB_API}/users/{org}/repos"
            resp = client.get(url, params=params)
            resp.raise_for_status()
            repos = resp.json()

        # Filter out forks and pick most starred
        non_forks = [r for r in repos if not r.get("fork", False)]
        if non_forks:
            return max(non_forks, key=lambda r: r.get("stargazers_count", 0))
        return repos[0] if repos else None
    except Exception as e:
        logger.error(f"Repo search failed for {org}: {e}")
        return None


def _count_commits_last_30d(client: httpx.Client, org: str, repo: str) -> int:
    """Count commits in the last 30 days using the commits API."""
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    url = f"{GITHUB_API}/repos/{org}/{repo}/commits"
    params = {"since": since, "per_page": 100}
    total = 0
    try:
        for page in range(1, 4):  # max 3 pages = 300 commits (enough signal)
            params["page"] = page
            resp = client.get(url, params=params)
            if resp.status_code == 409:  # empty repo
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            total += len(batch)
            if len(batch) < 100:
                break
    except Exception as e:
        logger.warning(f"Commit count failed for {org}/{repo}: {e}")
    return total


def _count_contributors(client: httpx.Client, org: str, repo: str) -> int:
    """Count unique contributors (uses stats endpoint, may be cached by GitHub)."""
    url = f"{GITHUB_API}/repos/{org}/{repo}/contributors"
    params = {"per_page": 100, "anon": "false"}
    total = 0
    try:
        for page in range(1, 6):
            params["page"] = page
            resp = client.get(url, params=params)
            if resp.status_code in (204, 202):  # 202 = stats being generated
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            total += len(batch)
            if len(batch) < 100:
                break
    except Exception as e:
        logger.warning(f"Contributor count failed: {e}")
    return total


def _get_release_cadence(client: httpx.Client, org: str, repo: str) -> tuple[Optional[float], Optional[str]]:
    """
    Returns (avg_days_between_releases, latest_release_tag).
    Uses last 10 releases to compute cadence.
    """
    url = f"{GITHUB_API}/repos/{org}/{repo}/releases"
    params = {"per_page": 10}
    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        releases = resp.json()
        if not releases:
            return None, None

        latest = releases[0].get("tag_name", "")
        if len(releases) < 2:
            return None, latest

        dates = []
        for r in releases:
            published = r.get("published_at")
            if published:
                dates.append(datetime.fromisoformat(published.replace("Z", "+00:00")))

        if len(dates) >= 2:
            gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            return round(avg_gap, 1), latest

        return None, latest
    except Exception as e:
        logger.warning(f"Release cadence failed: {e}")
        return None, None


def _compute_health_score(
    stars: int,
    commits_30d: int,
    contributors: int,
    open_issues: int,
    release_cadence: Optional[float],
) -> float:
    """
    Compute 0-10 engineering health score.
    Weights: commit velocity (35%), contributor base (25%),
             star traction (20%), issue management (10%),
             release discipline (10%).
    """
    # Commit velocity: 0-10 (100+ commits/30d = 10)
    commit_score = min(commits_30d / 10, 10)

    # Contributors: 0-10 (50+ = 10)
    contrib_score = min(contributors / 5, 10)

    # Stars: 0-10 (log scale, 10k+ = 10)
    import math
    star_score = min(math.log10(max(stars, 1)) * 2.5, 10)

    # Issue management: fewer is better (relative to stars)
    if stars > 0:
        issue_ratio = open_issues / max(stars, 1)
        issue_score = max(10 - issue_ratio * 100, 0)
    else:
        issue_score = 5.0

    # Release cadence: 0-10 (monthly = 10, never = 0)
    if release_cadence and release_cadence > 0:
        cadence_score = min(30 / release_cadence * 3.3, 10)
    else:
        cadence_score = 3.0

    weighted = (
        commit_score * 0.35
        + contrib_score * 0.25
        + star_score * 0.20
        + issue_score * 0.10
        + cadence_score * 0.10
    )
    return round(min(weighted, 10), 2)


async def fetch_github_signals(
    org: str,
    repo: Optional[str] = None,
) -> Optional[GitHubSignal]:
    """
    Main entry point for the GitHub agent.
    If repo is None, auto-discovers the most significant repo for the org.
    """
    logger.info(f"[GitHub Agent] Fetching signals for {org}/{repo or 'auto'}")

    headers = _make_headers()

    with httpx.Client(headers=headers, timeout=20, follow_redirects=True) as client:
        # Resolve repo if not given
        if not repo:
            repo_data = _find_main_repo(client, org)
            if not repo_data:
                logger.warning(f"No repos found for org: {org}")
                return None
            repo = repo_data["name"]
            logger.info(f"[GitHub Agent] Auto-selected repo: {org}/{repo}")
        else:
            # Fetch repo metadata directly
            url = f"{GITHUB_API}/repos/{org}/{repo}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                repo_data = resp.json()
            except Exception as e:
                logger.error(f"Repo metadata failed for {org}/{repo}: {e}")
                return None

        stars = repo_data.get("stargazers_count", 0)
        forks = repo_data.get("forks_count", 0)
        open_issues = repo_data.get("open_issues_count", 0)

        # Fetch languages
        lang_url = f"{GITHUB_API}/repos/{org}/{repo}/languages"
        try:
            lang_resp = client.get(lang_url)
            languages = lang_resp.json() if lang_resp.status_code == 200 else {}
        except Exception:
            languages = {}

        # Fetch commit count, contributors, cadence
        commits_30d = _count_commits_last_30d(client, org, repo)
        contributors = _count_contributors(client, org, repo)
        cadence, latest_release = _get_release_cadence(client, org, repo)

    health_score = _compute_health_score(
        stars, commits_30d, contributors, open_issues, cadence
    )

    logger.info(
        f"[GitHub Agent] {org}/{repo}: stars={stars}, commits_30d={commits_30d}, "
        f"contributors={contributors}, health={health_score}"
    )

    return GitHubSignal(
        org=org,
        repo=repo,
        stars=stars,
        forks=forks,
        open_issues=open_issues,
        commits_last_30d=commits_30d,
        contributors_count=contributors,
        top_languages=languages,
        release_cadence_days=cadence,
        last_release=latest_release,
        engineering_health_score=health_score,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import json

    logging.basicConfig(level=logging.INFO)

    async def test():
        # Test with a well-known fintech open-source org
        result = await fetch_github_signals("stripe", "stripe-python")
        if result:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print("No result")

    asyncio.run(test())
