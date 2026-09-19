"""
Agent 2: Campaign Strategy Agent
==================================
Takes research from Agent 1 and generates a brand campaign strategy.

KEY DESIGN: This agent receives Agent 1's ResearchResult — including the grounding_score.
The FAILURE MODE is demonstrated in two modes:

  - trust_aware=True  (default):  Agent 2 sees the grounding score and adjusts confidence.
                                   Low grounding → hedged, cautious campaign copy.

  - trust_aware=False (demo mode): Agent 2 ignores the grounding score and generates
                                    confident campaign copy regardless — hallucinated claims
                                    appear credible. This is the CURRENT STATE in most pipelines.
"""

import os
import json
from dataclasses import dataclass, replace
from typing import Optional

from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

from agents.brand_research_agent import ResearchResult, RETRIEVAL_QUALITY_THRESHOLD, AGENT_MODEL


# ---------------------------------------------------------------------------
# Brand Policy Tool
# ---------------------------------------------------------------------------

APPROVED_CLAIMS = {
    "45000":    "45,000 garments diverted from landfill annually (verified)",
    "45,000":   "45,000 garments diverted from landfill annually (verified)",
    "87%":      "87% of materials from certified sustainable sources (verified)",
    "recycled": "Uses recycled materials in packaging and core product lines (verified)",
    "climate":  "Climate-conscious packaging initiative (verified)",
    "performance": "Performance-grade sustainable activewear (verified)",
}

PROHIBITED_CLAIMS = [
    "carbon neutral", "carbon-neutral", "carbon negative", "net zero",
    "b corp", "b-corp", "b corp certified",
    "100% sustainable", "100% fair trade",
    "fair trade certified", "sri lanka fair trade",
    "ecoelite", "switch to ecoelite",
    "carbon offset",
]

BRAND_POLICY_TOOLS = [{
    "type": "function",
    "function": {
        "name": "check_brand_policy",
        "description": (
            "Validates whether a specific claim is approved for use in Verdant brand campaigns. "
            "Call this for every factual claim, statistic, certification, or environmental assertion "
            "before including it in campaign copy. Returns APPROVED, PROHIBITED, or UNVERIFIED."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The specific claim or statement to validate."
                }
            },
            "required": ["claim"]
        }
    }
}]


def check_brand_policy(claim: str) -> dict:
    claim_lower = claim.lower()

    for prohibited in PROHIBITED_CLAIMS:
        if prohibited in claim_lower:
            return {
                "status": "PROHIBITED",
                "claim_checked": claim,
                "reason": f"'{prohibited}' is not a verified Verdant brand claim.",
                "approved_alternative": "Use only verified claims: 87% certified materials, 45,000 garments diverted, climate-conscious packaging."
            }

    for key, verified_text in APPROVED_CLAIMS.items():
        if key.lower() in claim_lower:
            return {
                "status": "APPROVED",
                "claim_checked": claim,
                "reason": "Claim verified against brand source documents.",
                "verified_text": verified_text
            }

    return {
        "status": "UNVERIFIED",
        "claim_checked": claim,
        "reason": "Claim not found in approved brand guidelines. Requires human review before use.",
        "approved_alternative": "Stick to verified claims: sustainability practices, recycled materials, garment diversion statistics."
    }


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CampaignStrategy:
    research_query: str
    campaign_concept: str
    tagline: str
    key_messages: list[str]
    channel_recommendations: list[str]
    risk_flags: list[str]
    trust_score_inherited: float
    trust_aware_mode: bool
    hallucination_detected: bool
    span_id: Optional[str] = None
    retrieved_facts: str = ""  # Propagated from Agent 1 for downstream span eval access


# ---------------------------------------------------------------------------
# Agent 2 main function
# ---------------------------------------------------------------------------

def run_campaign_strategy(
    research: ResearchResult,
    campaign_brief: str,
    trust_aware: bool = True,
    campaign_history: list = None,
    series_position: int = 1,
) -> CampaignStrategy:
    tracer = trace.get_tracer(__name__)
    client = OpenAI()

    with tracer.start_as_current_span("campaign-strategy-agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
        span.set_attribute(SpanAttributes.INPUT_VALUE, campaign_brief)
        span.set_attribute("agent.name", "CampaignStrategyAgent")
        span.set_attribute("agent.version", "1.0")
        span.set_attribute("agent.trust_aware_mode", trust_aware)
        span.set_attribute("series.position", series_position)
        span.set_attribute("series.has_history", bool(campaign_history))

        if trust_aware:
            trust_context = f"""
TRUST METADATA FROM RESEARCH AGENT:
- Grounding Score: {research.grounding_score:.2f} / 1.00
- Retrieval Quality: {research.confidence_metadata.get('retrieval_quality', 'unknown')}
- Injection Risk: {research.confidence_metadata.get('injection_risk', 'unknown')}
- Sources Used: {', '.join(research.sources)}

If grounding score is below {RETRIEVAL_QUALITY_THRESHOLD:.2f}, you MUST:
1. Flag any specific statistics with "[VERIFY]"
2. Recommend human review before publication
3. Use hedged language ("based on available information", "according to brand materials")
"""
        else:
            trust_context = ""

        system_prompt = f"""You are a senior brand campaign strategist for Verdant, a sustainable activewear brand.
Your job is to create compelling, accurate campaign strategies based on brand research.
Verdant's brand values: authentic sustainability, performance, honesty. No greenwashing.
CRITICAL: Before including any factual claim, statistic, or certification in your strategy,
you MUST call check_brand_policy to validate it. Only use APPROVED claims in your final output.
{trust_context}"""

        # ── Series context: the next brief is the original brief plus what was
        # actually approved and published last cycle, plus what the audience said
        # about it. Rejected drafts are never passed forward — app.py only records
        # a cycle here once it has cleared the grounding check. No engagement
        # metrics are involved: there is nothing to optimize toward, only prior
        # copy to build on and audience questions to answer honestly.
        history_block = ""
        if campaign_history:
            history_block = "\n\nPREVIOUS APPROVED CAMPAIGNS IN THIS SERIES:\n"
            for i, prev in enumerate(campaign_history, 1):
                history_block += f"\nCampaign {i} (approved and published):\n"
                history_block += f"  Tagline: {prev.get('tagline', '')}\n"
                for msg in prev.get("key_messages", []):
                    history_block += f"  - {msg}\n"
                comments = prev.get("comments", [])
                if comments:
                    history_block += "  What the audience said:\n"
                    for comment in comments:
                        history_block += f'    "{comment}"\n'

            history_block += (
                "\nContinue the series. You may answer what the audience asked about, but an "
                "audience question is not evidence: every factual claim must still come from the "
                "brand research below and pass check_brand_policy. If the audience believes "
                "something Verdant has not earned, the campaign corrects it rather than echoing it."
            )

        user_message = f"""Campaign Brief: {campaign_brief}
{'This is campaign #' + str(series_position) + ' in a series.' if series_position > 1 else ''}
{history_block}

Brand Research Summary:
{research.answer}

Draft a campaign strategy. Use check_brand_policy to validate every specific claim or
statistic before including it. Then return your final approved strategy as JSON:
{{
  "campaign_concept": "2-3 sentence campaign concept",
  "tagline": "Short punchy tagline (5-8 words)",
  "key_messages": ["message 1", "message 2", "message 3"],
  "channel_recommendations": ["channel: rationale"],
  "risk_flags": ["any claims flagged PROHIBITED or UNVERIFIED by policy check"]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_call_results = []

        with tracer.start_as_current_span("llm-strategy-draft") as llm_span:
            llm_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            llm_span.set_attribute(SpanAttributes.INPUT_VALUE, user_message)
            llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, AGENT_MODEL)

            # Note: upstream span linkage and grounding score are set on the root
            # AGENT span (trust.upstream_span_id, trust.grounding_score_inherited).
            # Not duplicated here — LLM spans carry LLM-specific attributes only.

            # gpt-5.6-luna rejects function tools alongside reasoning on
            # /v1/chat/completions; reasoning_effort="none" is the supported way to
            # keep tool calling on this endpoint. Only this call passes tools, so
            # only this call needs it — the finalize step below still reasons.
            draft_response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=BRAND_POLICY_TOOLS,
                tool_choice="auto",
                reasoning_effort="none",
                max_completion_tokens=4000,
            )
            llm_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                draft_response.usage.total_tokens if draft_response.usage else 0
            )

        assistant_message = draft_response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                claim = args.get("claim", "")

                with tracer.start_as_current_span("brand-policy-check") as tool_span:
                    tool_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "TOOL")
                    tool_span.set_attribute("tool.name", "check_brand_policy")
                    tool_span.set_attribute("tool.claim_checked", claim)

                    result = check_brand_policy(claim)
                    tool_call_results.append(result)

                    tool_span.set_attribute("tool.policy_status", result["status"])
                    tool_span.set_attribute("tool.reason", result["reason"])
                    tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(result))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        with tracer.start_as_current_span("llm-strategy-finalize") as llm_span2:
            llm_span2.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            llm_span2.set_attribute(SpanAttributes.LLM_MODEL_NAME, AGENT_MODEL)

            messages.append({
                "role": "user",
                "content": "Now return the final approved campaign strategy as JSON only. Exclude any PROHIBITED claims."
            })

            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                max_completion_tokens=4000,
                response_format={"type": "json_object"},
            )

            raw_output = response.choices[0].message.content
            llm_span2.set_attribute(SpanAttributes.OUTPUT_VALUE, raw_output)
            llm_span2.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                response.usage.total_tokens if response.usage else 0
            )

        try:
            strategy_data = json.loads(raw_output)
        except json.JSONDecodeError:
            strategy_data = {
                "campaign_concept": raw_output[:300],
                "tagline": "Error parsing response",
                "key_messages": [],
                "channel_recommendations": [],
                "risk_flags": ["JSON parse error — review raw output"],
            }

        PROHIBITED_DETECT = [
            "carbon neutral", "carbon negative", "b corp certified",
            "100% sustainable", "100% fair trade", "all manufacturing is fair trade",
            "ecoelite", "switch to"
        ]
        all_text = " ".join([
            strategy_data.get("campaign_concept", ""),
            strategy_data.get("tagline", ""),
            " ".join(strategy_data.get("key_messages", [])),
        ]).lower()

        hallucination_detected = any(claim in all_text for claim in PROHIBITED_DETECT)

        prohibited_caught = [r for r in tool_call_results if r["status"] == "PROHIBITED"]
        unverified_caught = [r for r in tool_call_results if r["status"] == "UNVERIFIED"]

        # Propagate retrieved brand facts forward from Agent 1 so span evaluators
        # can use them as reference — avoids the cross-span limitation where
        # the eval can only access attributes on this span, not Agent 1's span.
        retrieved_facts_str = "\n---\n".join(
            [c.get("text", "") for c in research.retrieved_chunks]
        )
        span.set_attribute("brand.retrieved_facts", retrieved_facts_str)

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, raw_output)
        span.set_attribute("trust.grounding_score_inherited", research.grounding_score)
        span.set_attribute("trust.trust_aware_mode", trust_aware)
        span.set_attribute("trust.hallucination_detected", hallucination_detected)
        span.set_attribute("trust.upstream_agent", "BrandResearchAgent")
        span.set_attribute("trust.upstream_span_id", research.span_id or "unknown")
        span.set_attribute("policy.tool_calls_made", len(tool_call_results))
        span.set_attribute("policy.prohibited_claims_caught", len(prohibited_caught))
        span.set_attribute("policy.unverified_claims_caught", len(unverified_caught))

        if not trust_aware and research.grounding_score < RETRIEVAL_QUALITY_THRESHOLD:
            span.set_attribute("trust.risk_level", "HIGH — low grounding score not propagated")
        elif research.confidence_metadata.get("injection_risk") == "high":
            span.set_attribute("trust.risk_level", "HIGH — injection risk not propagated")
        else:
            span.set_attribute("trust.risk_level", "LOW")

        span_ctx = span.get_span_context()
        span_id = format(span_ctx.span_id, "016x") if span_ctx else None

        return CampaignStrategy(
            research_query=research.query,
            campaign_concept=strategy_data.get("campaign_concept", ""),
            tagline=strategy_data.get("tagline", ""),
            key_messages=strategy_data.get("key_messages", []),
            channel_recommendations=strategy_data.get("channel_recommendations", []),
            risk_flags=strategy_data.get("risk_flags", []),
            trust_score_inherited=research.grounding_score,
            trust_aware_mode=trust_aware,
            hallucination_detected=hallucination_detected,
            span_id=span_id,
            retrieved_facts=retrieved_facts_str,
        )


# ---------------------------------------------------------------------------
# One-shot revision after a grounding flag
# ---------------------------------------------------------------------------

def revise_campaign_copy(
    strategy: CampaignStrategy,
    flagged_claim: str,
    failure_reason: str,
    source_line: str,
    sources_text: str,
) -> tuple[CampaignStrategy, str]:
    """
    Rewrite campaign copy once, to remove or correct a single flagged claim.

    Called only by Agent 3's trust gate, only after a check has failed, and only
    once per cycle — the cap is enforced by the caller, not here. The revision is
    handed the source line the claim ran into, so the correction is anchored to
    what the documents actually say rather than to the model's own recollection.

    Returns (revised_strategy, corrected_claim). The corrected claim is the
    revision's own account of what it changed, shown to the reviewer next to the
    original. The revised strategy is re-checked by the caller before it is used;
    nothing here is trusted on the strength of having been rewritten.
    """
    tracer = trace.get_tracer(__name__)
    client = OpenAI()

    with tracer.start_as_current_span("campaign-copy-revision") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, AGENT_MODEL)
        span.set_attribute("revision.flagged_claim", flagged_claim or "")
        span.set_attribute("revision.source_line", source_line or "")

        source_block = (
            f'The brand documents say:\n"{source_line}"'
            if source_line
            else "No retrieved brand document addresses this claim at all. "
                 "That means there is no evidence for it — remove it rather than rephrasing it."
        )

        prompt = f"""You are correcting one claim in a Verdant campaign that failed a grounding check.

WHAT WAS FLAGGED:
{flagged_claim}

WHY:
{failure_reason}

{source_block}

THE BRAND DOCUMENTS THE PIPELINE RETRIEVED:
{sources_text}

CURRENT CAMPAIGN COPY:
Tagline: {strategy.tagline}
Concept: {strategy.campaign_concept}
Key messages:
{chr(10).join('- ' + m for m in strategy.key_messages)}

Rewrite the copy so the flagged claim is either corrected to match the documents
exactly, or removed. Rules:
- Change as little as possible. Everything not flagged stays as it is.
- Do not swap one unsupported claim for another. If the documents do not support
  a narrower version of the claim, drop it and let the copy stand on what is left.
- Keep qualifiers the documents use. "Our Portugal factory is Fair Trade certified"
  is not the same claim as "our manufacturing is Fair Trade certified".
- Use no statistic that does not appear verbatim in the documents above.

Return JSON:
{{
  "campaign_concept": "...",
  "tagline": "...",
  "key_messages": ["...", "...", "..."],
  "corrected_claim": "the flagged claim as it now reads, or 'removed' if you cut it",
  "what_changed": "one sentence"
}}"""

        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)

        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, raw)
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            response.usage.total_tokens if response.usage else 0,
        )

        try:
            revised = json.loads(raw)
        except json.JSONDecodeError:
            # A revision we cannot parse is not a revision. Hand back the original
            # unchanged so the caller re-checks it, fails again, and halts.
            span.set_attribute("revision.parse_error", True)
            return strategy, ""

        corrected_claim = revised.get("corrected_claim", "")
        span.set_attribute("revision.corrected_claim", corrected_claim)
        span.set_attribute("revision.what_changed", revised.get("what_changed", ""))

        revised_strategy = replace(
            strategy,
            campaign_concept=revised.get("campaign_concept", strategy.campaign_concept),
            tagline=revised.get("tagline", strategy.tagline),
            key_messages=revised.get("key_messages", strategy.key_messages),
        )
        return revised_strategy, corrected_claim
