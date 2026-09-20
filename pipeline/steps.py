"""
Pipeline steps. Fixed control flow — the model never decides what happens next.
===============================================================================

GENERATE → VERIFY → (REPAIR → RE-VERIFY) → ROUTE → ASSETS

Each step is a plain function taking data and returning data. Ordering lives in
run.py, not in any model's output.
"""

import os
import re
import json
import time
import random
from dataclasses import dataclass, field, replace
from typing import Optional

from openai import OpenAI

from brand_policy import check_brand_policy
from evals.grounding_check import (
    LAYER_NONE,
    TracedSpans,
    find_contradicting_source,
    run_grounding_check,
)
from pipeline.pricing import CostMeter, StepCost, cost_from_response, price_usage
from pipeline.usage_capture import capture_openai_usage, sum_usage  # noqa: F401 (re-exported for run.py)

GENERATE_MODEL = "gpt-5.6-luna"

# Agent 2's old prohibited-phrase heuristic caught two strings that
# check_brand_policy does not. There is no Agent 2 any more, so the check moves
# here rather than being silently dropped.
EXTRA_PROHIBITED = ["all manufacturing is fair trade", "switch to"]

VERDANT_WRITING_DIRECTION = """HOW TO WRITE IT — this is about the writing, not the facts:

The feeling is relief, not aspiration. Verdant's brand is honesty about an
incomplete job: the 13% is as much the story as the 87%. Write toward the sense
that someone has finally told you the truth about what they make.

- Use rhythm and concrete images. One vivid detail beats three statistics.
- The caption should be something a person stops scrolling for.
- Lead with the feeling and let the facts support it — not the other way round.
- Voice, rhythm, imagery and structure are NOT claims. They are not constrained
  by the source documents and you should use them freely.
- Only the facts need to trace. Everything else is yours."""

VERDANT_VOICE_RULES = """Verdant brand voice:
- Honest and direct. Never greenwash or exaggerate impact.
- Energetic but grounded. Performance without pretense.
- Community-first. Use "we", never "I".

Prohibited:
- The word "eco-friendly" (vague and overused)
- Any claim of carbon neutrality — Verdant is working toward it, not there
- Naming competitors directly
- Superlatives like "most sustainable" or "greenest" without data"""


# ---------------------------------------------------------------------------
# GENERATE
# ---------------------------------------------------------------------------

CAMPAIGN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tagline", "campaign_concept", "key_messages", "caption", "hashtags", "claims"],
    "properties": {
        "tagline": {"type": "string", "description": "5-8 words"},
        "campaign_concept": {"type": "string", "description": "2-3 sentences"},
        "key_messages": {"type": "array", "items": {"type": "string"}},
        "caption": {"type": "string", "description": "2-3 sentence social caption"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "claims": {
            "type": "array",
            "description": "Every verifiable factual claim made anywhere in this campaign.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "kind", "source_quote"],
                "properties": {
                    "text": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["statistic", "certification", "partnership",
                                 "material", "capability", "date", "program_result"],
                    },
                    "source_quote": {
                        "type": "string",
                        "description": "Verbatim line from the source documents supporting this claim.",
                    },
                },
            },
        },
    },
}


@dataclass
class CampaignDraft:
    tagline: str = ""
    campaign_concept: str = ""
    key_messages: list = field(default_factory=list)
    caption: str = ""
    hashtags: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    raw: str = ""

    def verified_text(self) -> str:
        """
        Everything publishable, which is what the guardrail checks. The caption
        and hashtags are included because they are literally what goes out —
        leaving them unchecked would ship unverified copy.
        """
        parts = [
            self.tagline,
            self.campaign_concept,
            " ".join(self.key_messages or []),
            self.caption,
            " ".join(self.hashtags or []),
        ]
        return "\n".join(p for p in parts if p and p.strip()).strip()


def format_corrections(corrections: list) -> str:
    """
    Corrections earned in earlier cycles of this series.

    Every run already records what was flagged and the source line it collided
    with; nothing used to read it afterwards, so the series relearned the same
    mistake every cycle. Each entry carries the claim AND the line that contradicted
    it, because knowing only what was wrong tells the model nothing about why.
    """
    if not corrections:
        return ""
    block = "\n\nALREADY CORRECTED EARLIER IN THIS SERIES — do not repeat these:\n"
    for c in corrections:
        block += f"\n  Cycle {c['cycle']} claimed: {c['claim']}\n"
        if c.get("source_line"):
            block += f"    The documents say: \"{c['source_line']}\"\n"
        if c.get("correction"):
            block += f"    Corrected to: {c['correction']}\n"
    block += ("\nThese are settled. Do not re-make any of them, and do not make the same "
              "kind of mistake with a different fact — if a document qualifies something, "
              "carry the qualifier.")
    return block


def _build_generate_prompt(brief: str, sources_text: str, audience_comments: list,
                           previous_copy: Optional[dict], cycle: int,
                           prior_corrections: list = None) -> str:
    """
    Build the GENERATE prompt.

    NOTE ON "sustainable" IN THE FIRST LINE — deliberate, do not "fix" it.
    BANNED_VISUAL_TERMS (further down this module) forbids words like "sustainable",
    "recycled" and "transparency", and that list applies to the VIDEO PROMPT ONLY.
    It exists because a video model renders an abstract noun as a literal object:
    asked for "recycled materials" it produced a shot of someone running past a
    recycling bin. Those words are hazards for a camera, not for a copywriter.

    Here they are just the product category and the subject matter; removing them
    would make the brief vaguer for no gain. Copy is constrained by the claim rules
    and by the guardrail, not by the visual vocabulary.

    The two constraints are intentionally asymmetric. Applying the video ban list
    to this prompt would be a regression, not a cleanup.
    """
    # Audience comments carry two different signals and drive two different
    # decisions. Read only for risk — as this block used to be — the series loses
    # the thing that makes it a series: each campaign ignores what the last one
    # actually landed. Read only for engagement, the campaign drifts toward
    # whatever the audience already believes, which is how unearned claims get in.
    # Subject matter comes from enthusiasm; claims come from the documents.
    comment_block = ""
    if audience_comments:
        joined = "\n".join(f'  "{c}"' for c in audience_comments)
        comment_block = (
            f"\n\nWHAT THE AUDIENCE SAID ABOUT THE LAST CAMPAIGN:\n{joined}\n"
            "\nRead these two ways and use both.\n"
            "\n1. WHAT THEY RESPONDED TO — the themes, angles and details they engaged\n"
            "   with. This decides what this campaign is ABOUT. Follow their interest and\n"
            "   go a layer deeper rather than repeating the last campaign: if they\n"
            "   responded to the materials story, this campaign is a more specific\n"
            "   materials story — the process, the supplier, the numbers behind it.\n"
            "\n2. WHAT THEY BELIEVE THAT IS NOT TRUE — assumptions Verdant has not earned.\n"
            "   This decides what you must NOT claim. Where the source documents let you,\n"
            "   correct the record plainly and in the campaign's own voice. Never repeat\n"
            "   the assumption back to them, even approvingly, and never soften it into a\n"
            "   hint that it might be true.\n"
            "\nEnthusiasm is a signal about subject matter, never about facts. An audience\n"
            "asking whether something is true is not a source saying that it is."
        )

    previous_block = ""
    if previous_copy:
        previous_block = (
            f"\n\nLAST APPROVED CAMPAIGN IN THIS SERIES:\n"
            f"  Tagline: {previous_copy.get('tagline', '')}\n"
            + "".join(f"  - {m}\n" for m in previous_copy.get("key_messages", []))
        )

    corrections_block = format_corrections(prior_corrections)

    return f"""You are writing a social campaign for Verdant, a sustainable activewear brand.

CAMPAIGN BRIEF: {brief}
{'This is campaign #' + str(cycle) + ' in a series.' if cycle > 1 else ''}
{previous_block}{comment_block}

{VERDANT_VOICE_RULES}

{VERDANT_WRITING_DIRECTION}

SOURCE DOCUMENTS — the only evidence you may draw factual claims from:
{sources_text}{corrections_block}

Write the campaign. Then list, in `claims`, every verifiable factual claim the
campaign makes — numbers, certifications, partnerships, materials, factory
locations, programme results, dates — and for each one quote the line from the
source documents above that supports it, verbatim.

If you cannot find a verbatim supporting line for a claim, do not make the claim.
Marketing voice and aspiration are not claims and do not belong in the list."""


def generate(
    brief: str,
    retrieval,
    audience_comments: list = None,
    previous_copy: dict = None,
    cycle: int = 1,
    client: OpenAI = None,
    prior_corrections: list = None,
) -> tuple[CampaignDraft, StepCost, str]:
    """One model call. Structured output. Returns (draft, cost, prompt)."""
    client = client or OpenAI()
    prompt = _build_generate_prompt(
        brief, retrieval.sources_text, audience_comments or [], previous_copy, cycle,
        prior_corrections=prior_corrections,
    )

    response = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "campaign", "strict": True, "schema": CAMPAIGN_SCHEMA},
        },
        max_completion_tokens=6000,
    )
    raw = response.choices[0].message.content or "{}"
    cost = cost_from_response("generate", GENERATE_MODEL, response)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    draft = CampaignDraft(
        tagline=data.get("tagline", ""),
        campaign_concept=data.get("campaign_concept", ""),
        key_messages=data.get("key_messages", []),
        caption=data.get("caption", ""),
        hashtags=data.get("hashtags", []),
        claims=data.get("claims", []),
        raw=raw,
    )
    return draft, cost, prompt


# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    passed: bool
    layer: str
    failure_type: Optional[str]
    claim_flagged: str
    source_line: str
    reason: str
    judge_called: bool
    judge_label: Optional[str]
    judge_explanation: str
    cost: StepCost
    judge_prompt: str = ""      # exactly what phoenix sent, captured off the wire


def _extra_prohibited_hit(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in EXTRA_PROHIBITED)


def verify(draft: CampaignDraft, retrieval, tracer=None, span_kinds=None,
           step_name: str = "verify") -> VerifyResult:
    """
    The guardrail. Deterministic checks first, judge only on survivors.

    Calls run_grounding_check() rather than reimplementing the cascade — the
    ordering and the fail-closed branch exist in exactly one place. Child spans
    are injected via TracedSpans so the trace shows both layers without a second
    copy of the logic.
    """
    text = draft.verified_text()
    spans = TracedSpans(tracer, span_kinds) if tracer is not None else None

    with capture_openai_usage() as usages:
        result = run_grounding_check(
            text,
            retrieval.sources_text,
            upstream_hallucination_flag=_extra_prohibited_hit(text),
            spans=spans,
        )

    if not result.judge_called:
        # Deterministic catch — no model call was made, so zero is the true cost,
        # not an unpriceable one. Routing this through price_usage() with an empty
        # model name made it look unavailable and turned the whole run's total to
        # None, which is precisely the kind of silently-wrong figure this module
        # is supposed to prevent.
        cost = StepCost(step=step_name, model="none (deterministic)", usd=0.0)
    else:
        from evals.grounding_check import JUDGE_MODEL
        prompt_t, completion_t, reasoning_t, cached_t, measured = sum_usage(usages)
        cost = price_usage(
            step_name, JUDGE_MODEL, prompt_t, completion_t,
            reasoning_tokens=reasoning_t, cached_tokens=cached_t, measured=measured,
        )

    # The judge request is built inside phoenix, so the only faithful record of it
    # is the one intercepted from the actual call.
    judge_prompt = ""
    if result.judge_called:
        from pipeline.usage_capture import rendered_prompts
        parts = rendered_prompts(usages)
        judge_prompt = "\n\n".join(p["text"] for p in parts)

    source_line = (
        "" if result.passed
        else find_contradicting_source(
            result.claim_flagged, retrieval.sources_text, result.failure_type
        )
    )

    return VerifyResult(
        passed=result.passed,
        layer=result.layer,
        failure_type=result.failure_type,
        claim_flagged=result.claim_flagged or "",
        source_line=source_line,
        reason=result.reason,
        judge_called=result.judge_called,
        judge_label=result.judge_label,
        judge_explanation=result.judge_explanation or "",
        cost=cost,
        judge_prompt=judge_prompt,
    )


# ---------------------------------------------------------------------------
# REPAIR
# ---------------------------------------------------------------------------

def repair(
    draft: CampaignDraft,
    verdict: VerifyResult,
    retrieval,
    client: OpenAI = None,
) -> tuple[CampaignDraft, StepCost, str, str]:
    """
    One model call, only on failure. Rewrites against the contradicting source.
    Returns (repaired_draft, cost, prompt, corrected_claim).

    Nothing here is trusted for having been rewritten — run.py re-verifies the
    result before it can be routed anywhere.
    """
    client = client or OpenAI()

    source_block = (
        f'The brand documents say:\n"{verdict.source_line}"'
        if verdict.source_line
        else "No retrieved brand document addresses this claim at all. There is no "
             "evidence for it — remove it rather than rephrasing it."
    )

    prompt = f"""One claim in this Verdant campaign failed a grounding check.

WHAT WAS FLAGGED:
{verdict.claim_flagged}

WHY:
{verdict.reason}

{source_block}

THE SOURCE DOCUMENTS:
{retrieval.sources_text}

CURRENT CAMPAIGN:
Tagline: {draft.tagline}
Concept: {draft.campaign_concept}
Key messages:
{chr(10).join('- ' + m for m in draft.key_messages)}
Caption: {draft.caption}
Hashtags: {' '.join(draft.hashtags)}

Rewrite so the flagged claim is corrected to match the documents exactly, or removed.
- Change as little as possible. Everything not flagged stays as it is.
- Do not swap one unsupported claim for another. If the documents do not support a
  narrower version, drop it and let the copy stand on what is left.
- Keep the qualifiers the documents use. "Our Portugal factory is Fair Trade
  certified" is not the same claim as "our manufacturing is Fair Trade certified".
- Use no statistic that does not appear verbatim in the documents above.

Return the full campaign as json, same shape as before, plus a "corrected_claim"
field saying how the flagged claim now reads (or "removed" if you cut it)."""

    schema = json.loads(json.dumps(CAMPAIGN_SCHEMA))
    schema["properties"]["corrected_claim"] = {"type": "string"}
    schema["required"] = schema["required"] + ["corrected_claim"]

    response = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "campaign_repair", "strict": True, "schema": schema},
        },
        max_completion_tokens=6000,
    )
    raw = response.choices[0].message.content or "{}"
    cost = cost_from_response("repair", GENERATE_MODEL, response)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # An unparseable repair is not a repair. Hand back the original so the
        # re-verify fails again and the campaign is routed to a human.
        return draft, cost, prompt, ""

    repaired = CampaignDraft(
        tagline=data.get("tagline", draft.tagline),
        campaign_concept=data.get("campaign_concept", draft.campaign_concept),
        key_messages=data.get("key_messages", draft.key_messages),
        caption=data.get("caption", draft.caption),
        hashtags=data.get("hashtags", draft.hashtags),
        claims=data.get("claims", draft.claims),
        raw=raw,
    )
    return repaired, cost, prompt, data.get("corrected_claim", "")


# ---------------------------------------------------------------------------
# ROUTE
# ---------------------------------------------------------------------------

DEST_PUBLISH = "publish"
DEST_HUMAN_FLAGGED = "human_flagged"
DEST_HUMAN_SAMPLE = "human_sample"

STATUS_APPROVED = "APPROVED"
STATUS_FLAGGED = "FLAGGED"
STATUS_SAMPLED = "SAMPLED"

DEFAULT_SAMPLE_RATE = 0.10


@dataclass
class RouteResult:
    destination: str
    sampled: bool
    final_status: str
    publishes: bool
    receipts: list = field(default_factory=list)


def build_receipts(draft: CampaignDraft, retrieval) -> list:
    """
    Pair each claim the model declared with the source line supporting it, so an
    approved campaign carries its evidence rather than just a verdict.

    The model supplies `source_quote`; we independently locate the best-matching
    line in the retrieved documents. When the two disagree, or when nothing is
    found, that is visible rather than assumed.
    """
    receipts = []
    for claim in draft.claims or []:
        claim_text = claim.get("text", "")
        located = find_contradicting_source(claim_text, retrieval.sources_text)
        receipts.append({
            "claim": claim_text,
            "kind": claim.get("kind", ""),
            "model_quoted_source": claim.get("source_quote", ""),
            "located_source_line": located,
            "source_found": bool(located),
        })
    return receipts


def route(
    draft: CampaignDraft,
    verdict: VerifyResult,
    retrieval,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    rng: random.Random = None,
) -> RouteResult:
    """
    Plain function, no model.

    A flagged campaign goes to a human and does not publish. A passing campaign
    publishes — including when it is sampled. Sampling audits what normally
    happens, so a sampled item must take the normal path; holding it back would
    create a second behaviour and mean observing that instead. The review copy
    goes out in parallel, after the fact.
    """
    if not verdict.passed:
        return RouteResult(
            destination=DEST_HUMAN_FLAGGED,
            sampled=False,
            final_status=STATUS_FLAGGED,
            publishes=False,
            receipts=[],
        )

    rng = rng or random
    sampled = sample_rate > 0 and rng.random() < sample_rate

    return RouteResult(
        destination=DEST_HUMAN_SAMPLE if sampled else DEST_PUBLISH,
        sampled=sampled,
        final_status=STATUS_SAMPLED if sampled else STATUS_APPROVED,
        publishes=True,
        receipts=build_receipts(draft, retrieval),
    )


# ---------------------------------------------------------------------------
# AUDIENCE — simulated reaction to what actually published
# ---------------------------------------------------------------------------

AUDIENCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["comments"],
    "properties": {
        "comments": {
            "type": "array",
            "description": "Exactly three social media comments.",
            "items": {"type": "string"},
        }
    },
}


def _escalation(cycle: int) -> str:
    """
    How far the audience has run ahead of the facts by this point in the series.

    The drift is the point of the series: an audience that likes a brand starts
    crediting it with things it never claimed, and the next campaign has to decide
    whether to go along. Early cycles stay accurate so there is a baseline to drift
    from.
    """
    if cycle <= 1:
        return (
            "All three comments are ACCURATE. They react to what the campaign actually "
            "said — enthusiastic, but adding nothing that is not in the copy."
        )
    if cycle == 2:
        return (
            "TWO comments are accurate. ONE contains a plausible over-reach: the kind of "
            "thing this audience would now assume, based on what this campaign said, but "
            "that the campaign did not actually claim."
        )
    return (
        f"ONE comment is accurate. TWO contain plausible over-reaches — by campaign "
        f"{cycle} this audience has been hearing the brand's story for a while and has "
        f"started crediting it with more than it earned. The over-reaches should feel "
        f"like a natural next step from what the campaign said, not a wild invention."
    )


def generate_audience_comments(
    draft: CampaignDraft,
    cycle: int = 1,
    client: OpenAI = None,
) -> tuple[list, StepCost, str]:
    """
    Three comments reacting to THIS campaign's published copy.

    Replaces a hardcoded list that was indexed by cycle number, so a Take Back
    campaign drew comments about recycled materials. Comments now have to be about
    what the campaign actually said.

    Returns (comments, cost, prompt).
    """
    client = client or OpenAI()

    prompt = f"""Three people comment on this brand's social post. Write their comments.

THE POST THEY ARE REPLYING TO:
{draft.tagline}
{draft.campaign_concept}
{draft.caption}
{' '.join(draft.hashtags)}

WHO THEY ARE: 22-38, care about where their clothes come from, follow this brand
already. They write like people on social media — short, casual, lowercase is fine,
an emoji sometimes. Not marketers. Not reviewers.

{_escalation(cycle)}

WHAT AN OVER-REACH LOOKS LIKE — always anchored to THIS post's subject, never a
generic complaint:
- Rounding a qualified number up: the post says 87%, the comment treats it as
  basically all of it.
- Assuming a certification the post implied but never claimed.
- Generalising from one case to every case: one certified factory becomes all of
  them, one recycled fabric becomes the whole range.
- Hearing a goal as an achievement: "targeting 95% by 2027" becomes "they're at 95%".

Whatever this post is about, that is what the comments are about. If the post is
about the Take Back Program, the comments are about the Take Back Program — do not
drift to some other part of the brand.

Return json with a "comments" array of exactly three strings."""

    response = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "audience", "strict": True, "schema": AUDIENCE_SCHEMA},
        },
        max_completion_tokens=2000,
    )
    cost = cost_from_response("audience", GENERATE_MODEL, response)
    try:
        comments = json.loads(response.choices[0].message.content or "{}").get("comments", [])
    except json.JSONDecodeError:
        comments = []
    return [c for c in comments if isinstance(c, str)][:3], cost, prompt


# ---------------------------------------------------------------------------
# ASSETS
# ---------------------------------------------------------------------------

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
class AssetResult:
    video_url: Optional[str] = None
    video_bytes: Optional[bytes] = None
    video_prompt: str = ""
    error: Optional[str] = None
    cost: Optional[StepCost] = None


# Abstract concept words a video model renders literally. Asked for "recycled
# materials" it produces a recycling bin; asked for "transparency" it produces
# glass. None of these may reach the video model — they are ideas, not things a
# camera can point at.
#
# SCOPE: VIDEO PROMPTS ONLY. This list must never be applied to campaign copy.
# The GENERATE prompt opens with "a sustainable activewear brand" and that is
# correct — see the note in _build_generate_prompt. A word that is a hazard for
# a camera is ordinary vocabulary for a copywriter, and stripping these from the
# copy would make it vaguer while fixing nothing. The asymmetry is the design.
#
# "green" is deliberately absent: it is a colour in the brand palette and has a
# real filmable meaning.
BANNED_VISUAL_TERMS = [
    "sustainable", "sustainability", "sustainably",
    "recycled", "recycle", "recycling",
    "eco", "eco-friendly", "ecological",
    "circular", "circularity",
    "transparency", "transparent",
    "purpose", "purposeful",
    "ethical", "ethically", "responsible", "responsibly",
    "conscious", "consciously", "mission", "values", "impact",
    "carbon", "footprint", "planet", "environment", "environmental",
    "certified", "certification", "fair trade", "organic",
    "activewear", "brand", "campaign",
]

# What the theme actually looks like. The translation happens before prompting,
# never inside the video model's head.
VISUAL_TRANSLATIONS = """How themes become images — translate, never name the theme:
- material quality      → extreme close-up of knit texture, fibres catching light, fabric flexing
- circularity / reuse   → the same garment worn across seasons, folded, packed, worn again
- openness / honesty    → wide frame, unbroken natural daylight, clean uncluttered space
- performance           → a body mid-stride, breath, muscle, motion blur on the limbs
- care for materials    → hands smoothing fabric, a seam in shallow focus
- longevity             → worn-in cloth, softened edges, a garment that has clearly been used"""


def find_banned_terms(text: str) -> list:
    """Concept words that survived into a video prompt. Deterministic, no model."""
    lowered = (text or "").lower()
    return sorted({t for t in BANNED_VISUAL_TERMS
                   if re.search(rf"\b{re.escape(t)}\b", lowered)})


# A shot has a beat when something changes inside the five seconds. These are the
# words that carry a change; a prompt with none of them is describing a state.
TURN_CUES = [
    "snaps", "snap", "releases", "release", "lets go", "springs",
    "turns", "turn", "flips", "reveals", "reveal", "revealing",
    "drops", "falls", "lands", "opens", "unfolds", "lifts", "rises",
    "cut to", "then", "until", "as it", "before", "after",
    "catches", "settles", "stops", "pulls away", "steps into", "shifts",
    "loosens", "tightens", "slips", "peels", "closes", "breaks",
]


def looks_like_scaffold(text: str) -> bool:
    """
    The five parts are how the shot is thought about, not how it is written.
    Veo needs flowing description; a labelled list gets read as literal content.
    """
    lowered = (text or "").lower()
    labels = ["subject —", "subject-", "the product", "the turn", "camera —", "light —",
              "**subject", "**the", "**camera", "**light"]
    hits = sum(1 for l in labels if l in lowered)
    return hits >= 2 or bool(re.match(r"^\s*(1\.|\*\*|#)", text or ""))


def has_turn(text: str) -> bool:
    """Does the shot contain something that happens, rather than a state?"""
    lowered = (text or "").lower()
    return any(re.search(rf"\b{re.escape(c)}\b", lowered) for c in TURN_CUES)


FALLBACK_VIDEO_PROMPT = (
    "Close on two hands stretching a panel of knit fabric taut, the weave opening "
    "under tension — then they let go and it snaps back into shape. Camera holds at "
    "hand height, shallow depth of field, then pulls back to reveal a runner pulling "
    "the garment on and stepping out of frame. Low morning light, forest greens and "
    "warm earth tones, film grain. No text, no logos."
)


def build_video_prompt(draft: CampaignDraft, client: OpenAI = None) -> tuple[str, StepCost, list, str]:
    """
    Turn the campaign into a shot description containing only things a camera can
    photograph. Returns (prompt, cost, banned_terms_that_survived).

    The campaign copy is NOT passed through. A video model reads "recycled
    materials" as a recycling bin, so the theme is translated to physical nouns
    first and the concept words never reach it.
    """
    client = client or OpenAI()

    def ask(extra: str = "") -> tuple[str, StepCost]:
        prompt_input = f"""You are a cinematographer planning one 5-second shot.

WHAT THE SHOT SHOULD EVOKE (background only — these words must NOT appear in your output):
{draft.campaign_concept}
{' '.join(draft.key_messages[:2])}

{VISUAL_TRANSLATIONS}

BUILD THE SHOT FROM FIVE PARTS, IN THIS ORDER:
1. SUBJECT      — who or what is in frame
2. THE PRODUCT  — the garment, clearly visible and identifiable. Not implied, not
                  off-screen. If nobody can tell what is being sold, the shot failed.
3. THE TURN     — the thing that happens. MANDATORY. Five seconds is enough for one
                  turn: something changes, is released, is revealed, or is handed on.
                  "A person doing an activity" is NOT a shot. A state is not a turn.
4. CAMERA       — height, angle, movement, lens feel
5. LIGHT        — time of day, direction, quality

THE SHOT MUST MAKE THE CAMPAIGN'S POINT ON ITS OWN, with the sound off and the
caption hidden. If it only works once someone reads the caption, rewrite it.

SHAPES THAT WORK:
- Hands stretch the fabric taut and let go; it snaps back.
- A worn garment drops into a box; cut to the same garment on someone new.
- A tag is turned over, the certification visible, then the wearer moves off.

LOOK: deep forest greens, warm earth tones, clean whites. Outdoors, natural light.
Real people mid-movement, not models posing. Editorial, cinematic, natural texture.
No text, no logos, no studio backgrounds.

WRITE IT AS 2-4 SENTENCES OF FLOWING DESCRIPTION. The five parts above are how you
think about the shot, NOT how you write it. Do not label them, number them, or use
headings — a labelled list gets read as literal content by the video model.

Max 90 words. Physical, filmable content only — no ideas, no qualities, no brand language.

FORBIDDEN WORDS — using any of these fails the task:
{', '.join(BANNED_VISUAL_TERMS)}

Bad:  "A runner embodies sustainable performance." (no product, no turn, abstract)
Bad:  "A runner moves along a trail at dawn." (a state, not a turn)
Good: "Hands twist a damp legging hem; water beads scatter, then the fabric springs flat.
       Camera low, tracking back as she stands into stride. Side light, dawn."
{extra}
Return only the shot description."""

        response = client.chat.completions.create(
            model=GENERATE_MODEL,
            messages=[{"role": "user", "content": prompt_input}],
            max_completion_tokens=2000,
        )
        return ((response.choices[0].message.content or "").strip(),
                cost_from_response("assets.prompt", GENERATE_MODEL, response),
                prompt_input)

    text, cost, sent = ask()
    banned = find_banned_terms(text)
    missing_turn = not has_turn(text)
    scaffolded = looks_like_scaffold(text)

    if banned or missing_turn or scaffolded:
        # One retry, naming what leaked. Hard cap — no loop.
        notes = []
        if banned:
            notes.append(f"Your previous attempt used these forbidden words: "
                         f"{', '.join(banned)}. Replace each with what it physically "
                         f"looks like.")
        if scaffolded:
            notes.append("Your previous attempt came back as a labelled list. Rewrite it "
                         "as 2-4 sentences of flowing description with no labels, numbers "
                         "or headings.")
        if missing_turn:
            notes.append("Your previous attempt described a state, not a turn. Nothing "
                         "changed inside the five seconds. Give the shot one thing that "
                         "happens — something released, revealed, handed on, or snapping "
                         "back — and make sure the garment is clearly in frame.")
        text2, cost2, sent = ask("\n" + " ".join(notes))
        cost = price_usage(
            "assets.prompt", GENERATE_MODEL,
            cost.prompt_tokens + cost2.prompt_tokens,
            cost.completion_tokens + cost2.completion_tokens,
            reasoning_tokens=cost.reasoning_tokens + cost2.reasoning_tokens,
            cached_tokens=cost.cached_tokens + cost2.cached_tokens,
        )
        banned2 = find_banned_terms(text2)
        if banned2:
            # Still leaking a concept word. Ship a prompt that is known clean and
            # has its own turn, rather than one that renders the idea literally.
            return FALLBACK_VIDEO_PROMPT, cost, banned2, sent
        # A missing turn is a weaker shot, not an unsafe one — keep it and record it
        # rather than discarding a clean prompt for the generic fallback.
        return text2, cost, [], sent

    return text, cost, [], sent


def generate_video(video_prompt: str) -> AssetResult:
    """Veo call. Billed by Google, not by token — no USD attributed here."""
    try:
        from google import genai as google_genai
        from google.genai import types as google_types

        google_client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        operation = google_client.models.generate_videos(
            model="veo-3.1-lite-generate-preview",
            prompt=video_prompt,
            config=google_types.GenerateVideosConfig(number_of_videos=1),
        )

        waited, max_wait = 0, 120
        while not operation.done and waited < max_wait:
            time.sleep(10)
            waited += 10
            operation = google_client.operations.get(operation)

        if operation.done and operation.response.generated_videos:
            video = operation.response.generated_videos[0].video
            if getattr(video, "video_bytes", None):
                return AssetResult(video_bytes=video.video_bytes, video_prompt=video_prompt)
            if getattr(video, "uri", None):
                import requests as _requests
                dl = _requests.get(
                    video.uri,
                    headers={"x-goog-api-key": os.environ.get("GOOGLE_API_KEY", "")},
                    timeout=30,
                )
                dl.raise_for_status()
                return AssetResult(
                    video_bytes=dl.content, video_url=video.uri, video_prompt=video_prompt
                )
        return AssetResult(video_prompt=video_prompt, error="timeout")
    except Exception as e:
        return AssetResult(video_prompt=video_prompt, error=str(e))
