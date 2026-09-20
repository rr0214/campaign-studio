"""
Agent 3: Creative Execution Agent
===================================
Takes the approved campaign strategy from Agent 2 and produces a complete
social media creative package:
  - A 5-second Veo 3.1 Lite campaign video (Google Gen AI)
  - Social media caption + hashtags (GPT-4o-mini)

TRUST GATE: the campaign text is checked against the brand documents Agent 1
retrieved, before any creative assets are generated. Cheap checks run first:
prohibited phrases, unapproved statistics and competitor names are caught in
plain Python; only text that survives those reaches the LLM judge. This is the
"take action" principle — not just flagging bad output, but refusing to produce
creative assets from unverified brand claims.

Arize AX trace structure:
  creative-execution-agent (AGENT)
    ├── claim-grounding-check (CHAIN)  — deterministic checks, then LLM judge
    ├── prompt-engineering (LLM)       — builds brand-safe video prompt
    ├── veo-video-generation (TOOL)    — generates 5s social video via Veo 3.1 Lite
    └── caption-generation (LLM)      — writes social caption + hashtags
"""

import os
import time
import json
import base64
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

from agents.brand_research_agent import RETRIEVAL_QUALITY_THRESHOLD, AGENT_MODEL
from agents.campaign_strategy_agent import CampaignStrategy
from agents.campaign_strategy_agent import revise_campaign_copy
from evals.grounding_check import (
    LAYER_NONE,
    campaign_text_for_check,
    find_contradicting_source,
    run_grounding_check,
)

# Retrieval-quality cutoff, shared with Agents 1 and 2. This is now a REPORTING
# signal only. It no longer gates: the score is built from retrieval metrics and
# never saw the generated text, so a perfect retrieval plus a fabricating model
# scored high and sailed through. The gate below reads the text instead.
CRITICAL_TRUST_THRESHOLD = RETRIEVAL_QUALITY_THRESHOLD

# Verdant brand visual guidelines
VERDANT_VISUAL_GUIDELINES = """
Verdant brand visual identity:
- Color palette: deep forest greens, warm earth tones, clean whites
- Aesthetic: clean, natural, active, authentic — NOT corporate or stock-photo
- Setting: outdoor environments, urban parks, trails, natural light
- People: diverse, active, real — not models posing
- Mood: energetic but grounded, aspirational but honest
- Style: editorial photography aesthetic, cinematic, natural textures
- No text overlays, no logos, no studio backgrounds
"""


@dataclass
class CreativeResult:
    status: str                          # APPROVED | HALTED | ERROR
    campaign_concept: str
    tagline: str
    video_url: Optional[str]             # Veo generated video URL
    video_bytes: Optional[bytes]         # Raw video bytes if URL not available
    video_prompt: Optional[str]          # Prompt used to generate video
    caption: Optional[str]              # Social media caption
    hashtags: Optional[list]            # Suggested hashtags
    halt_reason: Optional[str]          # Why pipeline halted (if applicable)
    video_error: Optional[str]          # Actual error from Veo API if generation failed
    trust_score_inherited: float
    span_id: Optional[str] = None

    # ── Grounding gate record ────────────────────────────────────────────────
    # What the check saw, what it flagged, and whether the one allowed rewrite
    # fixed it. Populated on every run, halted or not, so a series can be read
    # cycle by cycle without re-deriving anything from the halt_reason string.
    original_draft: Optional[str] = None      # copy as Agent 2 first wrote it
    revised_draft: Optional[str] = None       # copy after the single rewrite, if one ran
    flagged_claim: Optional[str] = None       # the claim that tripped the first check
    source_line: Optional[str] = None         # the retrieved line it ran into
    corrected_claim: Optional[str] = None     # how the rewrite reworded it
    check_layer: Optional[str] = None         # deterministic | judge | none
    failure_type: Optional[str] = None        # PROHIBITED_PHRASE | UNAPPROVED_STAT | ...
    rewrite_attempted: bool = False
    rewrite_fixed: bool = False
    grounding_events: list = field(default_factory=list)  # one entry per attempt


def run_creative_execution(
    strategy: CampaignStrategy,
    brand_name: str = "Verdant",
) -> CreativeResult:
    tracer = trace.get_tracer(__name__)
    openai_client = OpenAI()

    with tracer.start_as_current_span("creative-execution-agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
        span.set_attribute(SpanAttributes.INPUT_VALUE, strategy.campaign_concept or "")
        span.set_attribute("agent.name", "CreativeExecutionAgent")
        span.set_attribute("agent.version", "1.0")
        span.set_attribute("agent.upstream_agent", "CampaignStrategyAgent")
        span.set_attribute("agent.upstream_span_id", strategy.span_id or "unknown")
        span.set_attribute("trust.grounding_score_inherited", strategy.trust_score_inherited)
        span.set_attribute("trust.hallucination_detected_upstream", strategy.hallucination_detected)
        # Reported, not gating — kept so weak retrieval stays visible in Arize
        # alongside whatever the grounding check decided about the text itself.
        span.set_attribute(
            "trust.retrieval_below_threshold",
            strategy.trust_score_inherited < CRITICAL_TRUST_THRESHOLD,
        )

        # ── TRUST GATE: claim grounding, with one revision allowed ────────────
        # Attempt 1 checks what Agent 2 wrote. If it fails, the copy is rewritten
        # once against the source line the claim ran into, and re-checked once.
        # Two attempts, hard cap, no loop: a model that cannot fix its own claim
        # in one try is not going to find its way there in five, and each pass
        # costs a judge call. Whatever the second check says is final.
        MAX_ATTEMPTS = 2

        grounding_events = []
        original_draft = campaign_text_for_check(strategy)
        revised_draft = None
        corrected_claim = ""
        first_failure = None

        campaign_text = original_draft
        check = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            with tracer.start_as_current_span("claim-grounding-check") as gate_span:
                gate_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
                gate_span.set_attribute(SpanAttributes.INPUT_VALUE, campaign_text)
                gate_span.set_attribute("trust.attempt", attempt)

                check = run_grounding_check(
                    campaign_text,
                    strategy.retrieved_facts,
                    upstream_hallucination_flag=(
                        strategy.hallucination_detected if attempt == 1 else False
                    ),
                )

                source_line = (
                    ""
                    if check.passed
                    else find_contradicting_source(
                        check.claim_flagged, strategy.retrieved_facts, check.failure_type
                    )
                )

                gate_span.set_attribute("trust.check_layer", check.layer)
                gate_span.set_attribute("trust.failure_type", check.failure_type or "")
                gate_span.set_attribute("trust.claim_flagged", check.claim_flagged or "")
                gate_span.set_attribute("trust.judge_called", check.judge_called)
                gate_span.set_attribute("trust.judge_label", check.judge_label or "")
                gate_span.set_attribute("trust.source_line", source_line)
                gate_span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE,
                    "PASSED" if check.passed else f"HALTED — {check.failure_type}",
                )

            grounding_events.append({
                "attempt": attempt,
                "passed": check.passed,
                "layer": check.layer if not check.passed else LAYER_NONE,
                "failure_type": check.failure_type or "",
                "claim_flagged": check.claim_flagged or "",
                "source_line": source_line,
                "judge_called": check.judge_called,
                "draft": campaign_text,
            })

            if check.passed:
                break

            if attempt == MAX_ATTEMPTS:
                break

            # Only the first failure is worth reporting to the human — it is the
            # claim the campaign actually started from.
            first_failure = check
            strategy, corrected_claim = revise_campaign_copy(
                strategy=strategy,
                flagged_claim=check.claim_flagged,
                failure_reason=check.reason,
                source_line=source_line,
                sources_text=strategy.retrieved_facts,
            )
            campaign_text = campaign_text_for_check(strategy)
            revised_draft = campaign_text

        rewrite_attempted = len(grounding_events) > 1
        rewrite_fixed = rewrite_attempted and check.passed

        # Mirror the final state onto the agent span so a cycle is one row in Arize.
        span.set_attribute("trust.check_layer", check.layer if not check.passed else LAYER_NONE)
        span.set_attribute("trust.failure_type", check.failure_type or "")
        span.set_attribute("trust.claim_flagged", check.claim_flagged or "")
        span.set_attribute("trust.judge_called", any(e["judge_called"] for e in grounding_events))
        span.set_attribute("trust.attempts", len(grounding_events))
        span.set_attribute("trust.rewrite_attempted", rewrite_attempted)
        span.set_attribute("trust.rewrite_fixed", rewrite_fixed)
        span.set_attribute(
            "trust.first_failure_type",
            grounding_events[0]["failure_type"] if grounding_events else "",
        )

        gate_detail = dict(
            original_draft=original_draft,
            revised_draft=revised_draft,
            flagged_claim=grounding_events[0]["claim_flagged"] or None,
            source_line=grounding_events[0]["source_line"] or None,
            corrected_claim=corrected_claim or None,
            check_layer=grounding_events[0]["layer"],
            failure_type=grounding_events[0]["failure_type"] or None,
            rewrite_attempted=rewrite_attempted,
            rewrite_fixed=rewrite_fixed,
            grounding_events=grounding_events,
        )

        if not check.passed:
            span.set_attribute("trust.pipeline_halted", True)
            span.set_attribute("trust.halt_reason", check.failure_type or "unsupported_claim")
            halt_reason = check.reason
            if rewrite_attempted:
                halt_reason = (
                    f"{check.reason} A revision was attempted and re-checked; it did not clear "
                    "the grounding check either. Both drafts are shown for review."
                )
            return _halted_result(strategy, halt_reason, span, **gate_detail)

        span.set_attribute("trust.pipeline_halted", False)

        # ── PHASE 1: Build brand-safe video prompt ────────────────────────────
        with tracer.start_as_current_span("prompt-engineering") as prompt_span:
            prompt_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            prompt_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, AGENT_MODEL)

            prompt_input = f"""You are a creative director for {brand_name}, a sustainable activewear brand.

Brand visual guidelines:
{VERDANT_VISUAL_GUIDELINES}

Campaign concept: {strategy.campaign_concept}
Tagline: {strategy.tagline}
Key messages: {', '.join(strategy.key_messages[:2])}

Write a cinematic video prompt for a 5-second social media campaign video (max 150 words):
- Describe a single continuous scene or motion
- Follow Verdant brand visual guidelines strictly
- No text, logos, or words in the video
- Cinematic, natural, authentic
- Suitable for Instagram Reels or TikTok

Return only the video prompt, nothing else."""

            prompt_span.set_attribute(SpanAttributes.INPUT_VALUE, prompt_input)

            prompt_response = openai_client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[{"role": "user", "content": prompt_input}],
                max_completion_tokens=2000,
            )

            video_prompt = prompt_response.choices[0].message.content.strip()
            prompt_span.set_attribute(SpanAttributes.OUTPUT_VALUE, video_prompt)

        # ── PHASE 2: Generate video with Veo 3.1 Lite ────────────────────────
        video_url = None
        video_bytes = None
        veo_error = None

        with tracer.start_as_current_span("veo-video-generation") as veo_span:
            veo_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "TOOL")
            veo_span.set_attribute("tool.name", "veo-3.1-lite-generate-preview")
            veo_span.set_attribute("tool.duration_seconds", 5)
            veo_span.set_attribute(SpanAttributes.INPUT_VALUE, video_prompt)

            try:
                from google import genai as google_genai
                from google.genai import types as google_types

                google_client = google_genai.Client(
                    api_key=os.environ.get("GOOGLE_API_KEY")
                )

                operation = google_client.models.generate_videos(
                    model="veo-3.1-lite-generate-preview",
                    prompt=video_prompt,
                    config=google_types.GenerateVideosConfig(
                        number_of_videos=1,
                    ),
                )

                max_wait = 120
                waited = 0
                while not operation.done and waited < max_wait:
                    time.sleep(10)
                    waited += 10
                    operation = google_client.operations.get(operation)

                if operation.done and operation.response.generated_videos:
                    generated_video = operation.response.generated_videos[0].video

                    if hasattr(generated_video, 'video_bytes') and generated_video.video_bytes:
                        video_bytes = generated_video.video_bytes
                    elif hasattr(generated_video, 'uri') and generated_video.uri:
                        uri = generated_video.uri
                        import requests as _requests
                        api_key = os.environ.get("GOOGLE_API_KEY", "")
                        dl = _requests.get(
                            uri,
                            headers={"x-goog-api-key": api_key},
                            timeout=30,
                        )
                        dl.raise_for_status()
                        video_bytes = dl.content
                        video_url = uri

                    veo_span.set_attribute("tool.status", "success")
                    veo_span.set_attribute(SpanAttributes.OUTPUT_VALUE, video_url or "video_bytes_generated")
                else:
                    veo_span.set_attribute("tool.status", "timeout")

            except Exception as e:
                veo_span.set_attribute("tool.status", "error")
                veo_span.set_attribute("tool.error", str(e))
                video_bytes = None
                video_url = None
                veo_error = str(e)

        # ── PHASE 3: Generate social caption ─────────────────────────────────
        with tracer.start_as_current_span("caption-generation") as caption_span:
            caption_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            caption_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, AGENT_MODEL)

            caption_input = f"""Write a social media caption for this campaign.

Brand: {brand_name} — sustainable activewear
Tagline: {strategy.tagline}
Key messages: {', '.join(strategy.key_messages[:3])}
Platform: Instagram / TikTok

Requirements:
- 2-3 sentences max
- Authentic, not corporate
- End with a call to action
- Include 5-7 relevant hashtags on a new line
- Only use verified brand claims — no greenwashing

Return caption then hashtags on separate line."""

            caption_span.set_attribute(SpanAttributes.INPUT_VALUE, caption_input)

            caption_response = openai_client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[{"role": "user", "content": caption_input}],
                max_completion_tokens=2000,
            )

            caption_raw = caption_response.choices[0].message.content.strip()
            caption_parts = caption_raw.split("\n\n")
            caption_text = caption_parts[0] if caption_parts else caption_raw
            hashtag_line = caption_parts[1] if len(caption_parts) > 1 else ""
            hashtags = [h.strip() for h in hashtag_line.split() if h.startswith("#")]

            caption_span.set_attribute(SpanAttributes.OUTPUT_VALUE, caption_raw)

        # ── Final span attributes ─────────────────────────────────────────────
        span.set_attribute("trust.pipeline_completed", True)
        span.set_attribute("brand.retrieved_facts", strategy.retrieved_facts or "")
        span.set_attribute("creative.video_generated", video_url is not None or video_bytes is not None)
        span.set_attribute("creative.model", "veo-3.1-lite-generate-preview")
        # Content attributes — visible in Arize for cross-run comparison and drift monitoring
        span.set_attribute("creative.tagline", strategy.tagline or "")
        span.set_attribute("creative.caption", caption_text or "")
        span.set_attribute("creative.hashtags", " ".join(hashtags) if hashtags else "")
        span.set_attribute("creative.video_prompt", video_prompt or "")
        # Output summary on root span — mirrors OpenInference convention
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{strategy.tagline} | {caption_text or ''}")

        return CreativeResult(
            status="APPROVED",
            campaign_concept=strategy.campaign_concept,
            tagline=strategy.tagline,
            video_url=video_url,
            video_bytes=video_bytes,
            video_prompt=video_prompt,
            caption=caption_text,
            hashtags=hashtags,
            halt_reason=None,
            video_error=veo_error,
            trust_score_inherited=strategy.trust_score_inherited,
            span_id=_get_span_id(span),
            **gate_detail,
        )


def _halted_result(strategy, halt_reason, span, **gate_detail) -> CreativeResult:
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, "PIPELINE_HALTED")
    return CreativeResult(
        status="HALTED",
        campaign_concept=strategy.campaign_concept,
        tagline=strategy.tagline,
        video_url=None,
        video_bytes=None,
        video_prompt=None,
        caption=None,
        hashtags=None,
        halt_reason=halt_reason,
        video_error=None,
        trust_score_inherited=strategy.trust_score_inherited,
        span_id=_get_span_id(span),
        **gate_detail,
    )


def _get_span_id(span) -> Optional[str]:
    ctx = span.get_span_context()
    return format(ctx.span_id, "016x") if ctx else None
