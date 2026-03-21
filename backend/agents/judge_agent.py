"""
VERTEX — Judge Agent
Scores each debate round and determines when confidence is sufficient
to produce a final verdict. Also generates the investment synthesis report.
"""
import json
import logging
import re
from typing import Optional

from ..config import get_llm
from langchain.schema import HumanMessage, SystemMessage

from ..config import settings
from ..models import DebateArgument, JudgeVerdict, DebateResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a neutral chief investment officer judging a debate
between a Bull analyst and a Bear analyst. You score arguments on:
- Evidence quality (is it backed by specific data?)
- Logical coherence (does the argument flow without contradictions?)
- Relevance (does it address the actual investment question?)
- Novelty (does it bring new information, or just repeat prior points?)

You are STRICTLY neutral. You do not favour bull or bear by default.
You reward whoever makes the stronger data-driven argument each round."""


async def score_debate_round(
    round_number: int,
    bull: DebateArgument,
    bear: DebateArgument,
) -> JudgeVerdict:
    """Score a single debate round and decide if confidence threshold is reached."""
    llm = get_llm(temperature=0.1)

    prompt = f"""Score this debate round between Bull and Bear analysts.

ROUND {round_number}

BULL ARGUMENT:
{bull.argument}

BEAR ARGUMENT:
{bear.argument}

Score each side 0-10 on evidence quality, logic, and relevance.
Determine if the cumulative debate has reached a clear investment conclusion.

Return ONLY valid JSON:
{{
  "bull_score": <0-10 float>,
  "bear_score": <0-10 float>,
  "strongest_bull_point": "<one sentence>",
  "strongest_bear_point": "<one sentence>",
  "confidence": <0.0-1.0, how decisive is the outcome so far>,
  "reasoning": "<2-3 sentences explaining the round scores>"
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        text = re.sub(r"```json|```", "", response.content).strip()
        data = json.loads(text)

        verdict = JudgeVerdict(
            round_number=round_number,
            bull_score=float(data.get("bull_score", 5.0)),
            bear_score=float(data.get("bear_score", 5.0)),
            strongest_bull_point=data.get("strongest_bull_point", ""),
            strongest_bear_point=data.get("strongest_bear_point", ""),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

        logger.info(
            f"[Judge] Round {round_number}: Bull={verdict.bull_score:.1f} "
            f"Bear={verdict.bear_score:.1f} Confidence={verdict.confidence:.2f}"
        )
        return verdict

    except Exception as e:
        logger.error(f"[Judge] Scoring failed: {e}")
        return JudgeVerdict(
            round_number=round_number,
            bull_score=5.0,
            bear_score=5.0,
            strongest_bull_point="Analysis unavailable",
            strongest_bear_point="Analysis unavailable",
            confidence=0.5,
            reasoning=f"Scoring error: {str(e)}",
        )


async def synthesise_final_verdict(
    ticker: str,
    company_name: str,
    rounds: list[dict],
    all_bull_args: list[DebateArgument],
    all_bear_args: list[DebateArgument],
    all_verdicts: list[JudgeVerdict],
) -> DebateResult:
    """
    After all debate rounds, produce the final investment verdict and full report.
    """
    # Compute cumulative scores
    total_bull = sum(v.bull_score for v in all_verdicts)
    total_bear = sum(v.bear_score for v in all_verdicts)
    num_rounds = len(all_verdicts)
    avg_bull = total_bull / num_rounds if num_rounds else 5.0
    avg_bear = total_bear / num_rounds if num_rounds else 5.0

    # Determine directional verdict
    diff = avg_bull - avg_bear
    if diff > 1.5:
        direction = "bullish"
    elif diff < -1.5:
        direction = "bearish"
    else:
        direction = "neutral"

    final_confidence = max(v.confidence for v in all_verdicts) if all_verdicts else 0.5
    overall_score = round((avg_bull + (10 - avg_bear)) / 2, 2)

    # Collect strongest points
    bull_points = [v.strongest_bull_point for v in all_verdicts if v.strongest_bull_point]
    bear_points = [v.strongest_bear_point for v in all_verdicts if v.strongest_bear_point]
    risk_factors = []
    for arg in all_bear_args:
        risk_factors.extend(arg.supporting_evidence)
    risk_factors = list(dict.fromkeys(risk_factors))[:5]  # dedupe, keep top 5

    # Final narrative synthesis
    llm = get_llm(temperature=0.3)

    bull_summary = "\n".join(f"- {p}" for p in bull_points)
    bear_summary = "\n".join(f"- {p}" for p in bear_points)

    synthesis_prompt = f"""Write a final investment verdict for {company_name} ({ticker}).

DEBATE OUTCOME:
- Direction: {direction.upper()}
- Bull score average: {avg_bull:.1f}/10
- Bear score average: {avg_bear:.1f}/10
- Overall score: {overall_score}/10

STRONGEST BULL POINTS:
{bull_summary}

STRONGEST BEAR POINTS:
{bear_summary}

Write:
1. INVESTMENT SUMMARY: 2-3 paragraph balanced synthesis
2. BULL CASE: 2-3 sentences
3. BEAR CASE: 2-3 sentences

Format with those exact headers."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a neutral chief investment officer writing a final report."),
            HumanMessage(content=synthesis_prompt),
        ])
        synthesis = response.content

        # Parse sections
        inv_match = re.search(r"INVESTMENT SUMMARY:(.*?)(?=BULL CASE:|$)", synthesis, re.DOTALL)
        bull_match = re.search(r"BULL CASE:(.*?)(?=BEAR CASE:|$)", synthesis, re.DOTALL)
        bear_match = re.search(r"BEAR CASE:(.*?)$", synthesis, re.DOTALL)

        investment_summary = inv_match.group(1).strip() if inv_match else synthesis
        bull_case = bull_match.group(1).strip() if bull_match else " ".join(bull_points[:2])
        bear_case = bear_match.group(1).strip() if bear_match else " ".join(bear_points[:2])

    except Exception as e:
        logger.error(f"[Judge] Synthesis failed: {e}")
        investment_summary = f"{direction.capitalize()} outlook with {overall_score:.1f}/10 score."
        bull_case = " ".join(bull_points[:2])
        bear_case = " ".join(bear_points[:2])

    logger.info(f"[Judge] Final verdict: {direction} ({overall_score}/10)")

    return DebateResult(
        ticker=ticker,
        company_name=company_name,
        rounds=rounds,
        final_verdict=direction,
        final_confidence=final_confidence,
        investment_summary=investment_summary,
        risk_factors=risk_factors,
        bull_case=bull_case,
        bear_case=bear_case,
        overall_score=overall_score,
    )
