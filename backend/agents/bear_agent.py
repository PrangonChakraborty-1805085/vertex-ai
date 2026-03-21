"""
VERTEX — Bear Analyst Agent
Challenges the bull case with data-driven counter-arguments.
Finds weaknesses, contradictions, and risks in the investment thesis.
"""
import logging
import re
from typing import Optional

from ..config import get_llm
from langchain.schema import HumanMessage, SystemMessage

from ..config import settings
from ..models import ResearchPackage, DebateArgument
from .bull_agent import _build_research_summary

logger = logging.getLogger(__name__)

BEAR_SYSTEM_PROMPT = """You are a skeptical short-seller and risk analyst at a hedge fund.
Your job is to find every flaw, risk, and overvaluation in the bull case.
You are NOT pessimistic for its own sake — you are data-driven and precise.
You attack weak assumptions, point out missing context, and identify structural risks.
You NEVER accept a bull point without challenging the underlying assumption.
When the Bull cites a metric, you question its quality, sustainability, or relevance.
You are sharp, contrarian, and difficult to argue with."""


async def run_bear_argument(
    package: ResearchPackage,
    round_number: int,
    bull_argument: Optional[str] = None,
    rag_context: Optional[str] = None,
) -> DebateArgument:
    """
    Generate a bear argument for a given round.
    Round 1: opening counter-thesis
    Round 2+: targeted rebuttal of bull's argument
    """
    research = _build_research_summary(package)

    rag_section = f"ADDITIONAL RAG CONTEXT:\n{rag_context}" if rag_context else ""

    if round_number == 1:
        user_prompt = f"""Make your OPENING BEAR CASE against investing in {package.company_name} ({package.ticker}).

RESEARCH DATA:
{research}

{rag_section}

Structure your argument with:
1. Core bear thesis (1 sentence)
2. Three strongest bear points — each must target a specific vulnerability in the data
3. Your confidence level (0.0 to 1.0)

Format:
THESIS: <one sentence>
POINTS:
- <risk/weakness 1 with data>
- <risk/weakness 2 with data>
- <risk/weakness 3 with data>
CONFIDENCE: <0.0-1.0>"""
    else:
        user_prompt = f"""The Bull has argued:
"{bull_argument}"

RESEARCH DATA:
{research}

Round {round_number}: Attack the Bull's argument.
- Identify the weakest assumption in their argument
- Present data that directly contradicts their strongest point
- Escalate the most serious structural risk they haven't addressed

Format:
THESIS: <your maintained/sharpened bear thesis>
POINTS:
- <targeted attack on their weakest claim>
- <contradicting data point>
- <structural risk they're ignoring>
CONFIDENCE: <0.0-1.0>"""

    llm = get_llm(temperature=0.4)

    try:
        response = llm.invoke([
            SystemMessage(content=BEAR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        content = response.content

        confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", content)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.6

        points = re.findall(r"^- (.+)$", content, re.MULTILINE)

        logger.info(f"[Bear Agent] Round {round_number} complete, confidence={confidence}")

        return DebateArgument(
            role="bear",
            round_number=round_number,
            argument=content,
            supporting_evidence=points[:3],
            confidence=min(max(confidence, 0.0), 1.0),
        )

    except Exception as e:
        logger.error(f"[Bear Agent] Failed: {e}")
        return DebateArgument(
            role="bear",
            round_number=round_number,
            argument=f"Bear analysis unavailable: {str(e)}",
            supporting_evidence=[],
            confidence=0.5,
        )
