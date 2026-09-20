"""
The pipeline. Fixed control flow, one linear pass.
==================================================

    campaign-run                  CHAIN   (root)
    ├── retrieve                  RETRIEVER
    ├── generate                  LLM
    ├── verify                    GUARDRAIL
    │   ├── deterministic-checks  TOOL
    │   └── grounding-judge       LLM      (only if deterministic passed)
    ├── repair                    LLM      (only on failure)
    ├── re-verify                 GUARDRAIL (only on failure)
    │   ├── deterministic-checks  TOOL
    │   └── grounding-judge       LLM
    ├── route                     CHAIN
    └── assets                    TOOL     (only when routing publishes)

Repair and re-verify are siblings, not nested: the re-check is the step after
the repair, not part of its model call, and a linear pipeline should read as
linear in the trace.

No step decides which step runs next. The only branches are on data — did
verification pass, did routing publish — never on a model's choice.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from pipeline.pricing import CostMeter, price_usage
from pipeline.retrieval import RETRIEVAL_QUALITY_THRESHOLD, derive_research_query, retrieve
from evals.grounding_check import JUDGE_MODEL as JUDGE_MODEL_NAME
from pipeline import steps as S


# GUARDRAIL is present in our openinference-semconv; the getattr keeps an older
# install from failing at import rather than at the first traced run.
_KIND = OpenInferenceSpanKindValues
GUARDRAIL_KIND = getattr(_KIND, "GUARDRAIL", _KIND.CHAIN).value
CHAIN_KIND = _KIND.CHAIN.value
LLM_KIND = _KIND.LLM.value
TOOL_KIND = _KIND.TOOL.value
RETRIEVER_KIND = _KIND.RETRIEVER.value

VERIFY_SPAN_KINDS = {"tool": TOOL_KIND, "llm": LLM_KIND}

MAX_REPAIRS = 1

# Where a run came from. Stamped on the root span so evaluation traffic can be
# filtered out of live metrics rather than averaged into them. Tracing is off in
# the eval path today; this is here so the day it is turned on, eval runs are
# already distinguishable instead of silently inflating live counts.
SOURCE_APP = "app"
SOURCE_EVAL = "eval"


@dataclass
class CampaignRun:
    """Everything one pass through the pipeline produced."""
    brief: str
    cycle: int
    source: str = SOURCE_APP

    retrieval: object = None
    draft: S.CampaignDraft = None
    original_draft: Optional[S.CampaignDraft] = None
    verdict: S.VerifyResult = None
    first_verdict: S.VerifyResult = None
    routing: S.RouteResult = None
    assets: S.AssetResult = None

    repair_attempted: bool = False
    repair_fixed: bool = False
    audience_comments_generated: list = field(default_factory=list)
    video_blocked: object = None   # GroundingCheckResult when the shot was refused
    video_judge_label: str = ""    # visual-claim judge verdict, when it ran
    video_concept_leaked: list = field(default_factory=list)  # forced the fallback shot
    corrected_claim: str = ""

    cost: CostMeter = field(default_factory=CostMeter)
    span_id: Optional[str] = None
    verify_events: list = field(default_factory=list)
    # The fully rendered text sent to a model, per call, in order. Not the
    # template — the actual string, so a given run can be read back exactly.
    prompts: list = field(default_factory=list)

    # ── Accessors the UI and evals read ──────────────────────────────────────
    @property
    def final_status(self) -> str:
        if self.video_blocked is not None:
            return "FLAGGED"
        return self.routing.final_status if self.routing else "ERROR"

    @property
    def published(self) -> bool:
        """
        One unit: a campaign is its copy AND its video. ROUTE settles before
        ASSETS runs, so a refused shot arrives after routing has already said
        publish — and it holds the whole campaign, exactly as a failed copy check
        does. Nothing ships in halves.
        """
        if self.video_blocked is not None:
            return False
        return bool(self.routing and self.routing.publishes)

    @property
    def flagged_claim(self) -> Optional[str]:
        return (self.first_verdict.claim_flagged or None) if self.first_verdict else None

    @property
    def source_line(self) -> Optional[str]:
        return (self.first_verdict.source_line or None) if self.first_verdict else None

    @property
    def check_layer(self) -> str:
        # A video block is the failure when the copy passed — reporting the copy
        # verdict's layer would say "none" on a campaign that was held.
        if self.video_blocked is not None and (
                not self.first_verdict or self.first_verdict.passed):
            return self.video_blocked.layer
        if not self.first_verdict:
            return LAYER_NONE_FALLBACK
        return self.first_verdict.layer if not self.first_verdict.passed else "none"

    @property
    def failure_type(self) -> Optional[str]:
        copy_failure = (self.first_verdict.failure_type or None) if self.first_verdict else None
        if copy_failure:
            return copy_failure
        # The audit record showed failure_type None beside status FLAGGED because
        # only the copy verdict was consulted.
        if self.video_blocked is not None:
            return self.video_blocked.failure_type
        return None

    def correction_record(self) -> Optional[dict]:
        """
        What this cycle learned, in the shape the next cycle's brief consumes.
        Only a repair that actually cleared the re-check counts as a correction —
        a flagged claim that was never fixed is not a lesson, it is an open issue.
        """
        if not self.first_verdict or self.first_verdict.passed:
            return None
        return {
            "cycle": self.cycle,
            "claim": self.first_verdict.claim_flagged,
            "source_line": self.first_verdict.source_line,
            "correction": self.corrected_claim if self.repair_fixed else "",
            "resolved": self.repair_fixed,
        }

    @property
    def halt_reason(self) -> Optional[str]:
        if self.video_blocked is not None:
            return self.video_blocked.reason
        if self.routing and self.routing.publishes:
            return None
        base = self.verdict.reason if self.verdict else None
        if base and self.repair_attempted:
            return (
                f"{base} A repair was attempted and re-checked; it did not clear the "
                "grounding check either. Both drafts are shown for review."
            )
        return base


LAYER_NONE_FALLBACK = "none"


def _notify(progress, step: str, state: str, detail: dict = None):
    """
    Report step transitions to a caller that wants to render progress (the UI).
    Purely observational — a progress callback can never change what runs next.
    """
    if progress is None:
        return
    try:
        progress(step, state, detail or {})
    except Exception:
        pass  # a broken progress display must not take the pipeline down


def _set_llm_span(span, model, prompt, output, cost):
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
    span.set_attribute(SpanAttributes.INPUT_VALUE, prompt or "")
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, output or "")
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, cost.prompt_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, cost.completion_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, cost.total_tokens)
    if cost.reasoning_tokens:
        span.set_attribute("llm.token_count.reasoning", cost.reasoning_tokens)
    if cost.cached_tokens:
        span.set_attribute("llm.token_count.cached", cost.cached_tokens)
    _set_step_cost(span, cost)


def _set_step_cost(span, cost):
    span.set_attribute("cost.step_tokens", cost.total_tokens)
    span.set_attribute("cost.pricing_tier", cost.pricing_tier)
    if cost.usd is not None:
        span.set_attribute("cost.step_usd", cost.usd)
    else:
        span.set_attribute("cost.step_usd_available", False)
    if not cost.measured:
        span.set_attribute("cost.measured", False)
    if cost.tier_exceeded:
        span.set_attribute("cost.tier_exceeded", True)


def _set_verify_span(span, verdict):
    span.set_attribute("trust.layer_reached", verdict.layer)
    span.set_attribute(
        "trust.deterministic_result",
        "FAILED" if (not verdict.passed and verdict.layer == "deterministic") else "PASSED",
    )
    span.set_attribute("trust.judge_verdict", verdict.judge_label or "not_called")
    span.set_attribute("trust.judge_explanation", verdict.judge_explanation or "")
    span.set_attribute("trust.judge_called", verdict.judge_called)
    span.set_attribute("trust.failure_type", verdict.failure_type or "")
    span.set_attribute("trust.claim_flagged", verdict.claim_flagged or "")
    span.set_attribute("trust.source_line", verdict.source_line or "")
    span.set_attribute(
        SpanAttributes.OUTPUT_VALUE,
        "PASSED" if verdict.passed else f"FAILED — {verdict.failure_type}",
    )
    _set_step_cost(span, verdict.cost)


def run_campaign_pipeline(
    brief: str,
    research_query: str = None,
    audience_comments: list = None,
    previous_copy: dict = None,
    prior_corrections: list = None,
    cycle: int = 1,
    include_poisoned: bool = False,
    simulate_low_confidence: bool = False,
    sample_rate: float = S.DEFAULT_SAMPLE_RATE,
    rng: random.Random = None,
    generate_assets: bool = True,
    simulate_audience: bool = False,
    client: OpenAI = None,
    progress=None,
    source: str = SOURCE_APP,
) -> CampaignRun:
    tracer = trace.get_tracer(__name__)
    client = client or OpenAI()
    run = CampaignRun(brief=brief, cycle=cycle, source=source)
    query = research_query or derive_research_query(brief)

    with tracer.start_as_current_span("campaign-run") as root:
        root.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, CHAIN_KIND)
        root.set_attribute(SpanAttributes.INPUT_VALUE, brief)
        root.set_attribute("campaign.brief", brief)
        root.set_attribute("campaign.cycle", cycle)
        root.set_attribute("run.source", source)

        # ── RETRIEVE ─────────────────────────────────────────────────────────
        _notify(progress, "retrieve", "start")
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, RETRIEVER_KIND)
            span.set_attribute(SpanAttributes.INPUT_VALUE, query)

            with S.capture_openai_usage() as embed_usages:
                run.retrieval = retrieve(
                    query,
                    include_poisoned=include_poisoned,
                    simulate_low_confidence=simulate_low_confidence,
                )
            p, c, r, cached, measured = S.sum_usage(embed_usages)
            embed_cost = run.cost.add(price_usage(
                "retrieve", "text-embedding-3-small", p, c,
                reasoning_tokens=r, cached_tokens=cached, measured=measured,
            ))

            span.set_attribute("retrieval.n_chunks", len(run.retrieval.chunks))
            span.set_attribute(
                SpanAttributes.RETRIEVAL_DOCUMENTS,
                json.dumps([
                    {"document": {"content": ch["text"], "metadata": {"source": ch["doc_name"]}},
                     "score": ch["relevance_score"]}
                    for ch in run.retrieval.chunks
                ]),
            )
            span.set_attribute("trust.n_chunks_retrieved", len(run.retrieval.chunks))
            span.set_attribute("trust.grounding_score", run.retrieval.grounding_score)
            span.set_attribute("trust.retrieval_quality",
                               run.retrieval.metadata["retrieval_quality"])
            span.set_attribute("trust.injection_risk", run.retrieval.metadata["injection_risk"])
            _set_step_cost(span, embed_cost)
        _notify(progress, "retrieve", "done", {"n_chunks": len(run.retrieval.chunks), "grounding_score": run.retrieval.grounding_score})

        # ── GENERATE ─────────────────────────────────────────────────────────
        _notify(progress, "generate", "start")
        with tracer.start_as_current_span("generate") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, LLM_KIND)
            run.draft, gen_cost, gen_prompt = S.generate(
                brief, run.retrieval, audience_comments, previous_copy, cycle,
                client=client, prior_corrections=prior_corrections,
            )
            run.cost.add(gen_cost)
            run.original_draft = run.draft
            run.prompts.append({"step": "generate", "model": S.GENERATE_MODEL,
                                "text": gen_prompt})
            _set_llm_span(span, S.GENERATE_MODEL, gen_prompt, run.draft.raw, gen_cost)
            span.set_attribute("campaign.n_claims", len(run.draft.claims))
            span.set_attribute("campaign.prior_corrections", len(prior_corrections or []))
        _notify(progress, "generate", "done", {"n_claims": len(run.draft.claims), "tagline": run.draft.tagline})

        # ── VERIFY ───────────────────────────────────────────────────────────
        _notify(progress, "verify", "start")
        with tracer.start_as_current_span("verify") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, GUARDRAIL_KIND)
            span.set_attribute(SpanAttributes.INPUT_VALUE, run.draft.verified_text())
            run.verdict = S.verify(
                run.draft, run.retrieval, tracer=tracer, span_kinds=VERIFY_SPAN_KINDS
            )
            run.cost.add(run.verdict.cost)
            run.first_verdict = run.verdict
            if run.verdict.judge_prompt:
                run.prompts.append({"step": "verify (judge)", "model": JUDGE_MODEL_NAME,
                                    "text": run.verdict.judge_prompt})
            _set_verify_span(span, run.verdict)
        _notify(progress, "verify", "done", {"passed": run.verdict.passed, "layer": run.verdict.layer, "failure_type": run.verdict.failure_type, "claim": run.verdict.claim_flagged, "judge_called": run.verdict.judge_called})

        run.verify_events.append(_event(1, run.verdict))

        # ── REPAIR + RE-VERIFY ───────────────────────────────────────────────
        # Bounded: exactly one repair, never a loop. A model that cannot fix its
        # own claim in one attempt will not find its way there in five, and each
        # pass costs a judge call.
        if not run.verdict.passed and MAX_REPAIRS:
            run.repair_attempted = True

            _notify(progress, "repair", "start")
            with tracer.start_as_current_span("repair") as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, LLM_KIND)
                span.set_attribute("trust.claim_flagged", run.verdict.claim_flagged or "")
                span.set_attribute("trust.source_line", run.verdict.source_line or "")
                repaired, rep_cost, rep_prompt, corrected = S.repair(
                    run.draft, run.verdict, run.retrieval, client=client
                )
                run.cost.add(rep_cost)
                run.corrected_claim = corrected
                run.prompts.append({"step": "repair", "model": S.GENERATE_MODEL,
                                    "text": rep_prompt})
                _set_llm_span(span, S.GENERATE_MODEL, rep_prompt, repaired.raw, rep_cost)
                span.set_attribute("repair.corrected_claim", corrected or "")
            _notify(progress, "repair", "done", {"corrected_claim": corrected})

            _notify(progress, "re-verify", "start")
            with tracer.start_as_current_span("re-verify") as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, GUARDRAIL_KIND)
                span.set_attribute(SpanAttributes.INPUT_VALUE, repaired.verified_text())
                second = S.verify(
                    repaired, run.retrieval, tracer=tracer,
                    span_kinds=VERIFY_SPAN_KINDS, step_name="re-verify",
                )
                run.cost.add(second.cost)
                if second.judge_prompt:
                    run.prompts.append({"step": "re-verify (judge)",
                                        "model": JUDGE_MODEL_NAME,
                                        "text": second.judge_prompt})
                _set_verify_span(span, second)
            _notify(progress, "re-verify", "done", {"passed": second.passed, "failure_type": second.failure_type, "claim": second.claim_flagged})

            run.verify_events.append(_event(2, second))
            run.draft = repaired
            run.verdict = second
            run.repair_fixed = second.passed

        # ── ROUTE ────────────────────────────────────────────────────────────
        _notify(progress, "route", "start")
        with tracer.start_as_current_span("route") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, CHAIN_KIND)
            run.routing = S.route(
                run.draft, run.verdict, run.retrieval, sample_rate=sample_rate, rng=rng
            )
            span.set_attribute("route.destination", run.routing.destination)
            span.set_attribute("route.sampled", run.routing.sampled)
            span.set_attribute("route.publishes", run.routing.publishes)
            span.set_attribute("route.n_receipts", len(run.routing.receipts))
            span.set_attribute(
                "route.receipts_without_source",
                sum(1 for r in run.routing.receipts if not r["source_found"]),
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, run.routing.final_status)
        _notify(progress, "route", "done", {"destination": run.routing.destination, "sampled": run.routing.sampled, "status": run.routing.final_status, "publishes": run.routing.publishes})

        # ── ASSETS ───────────────────────────────────────────────────────────
        # Sampled campaigns publish exactly like approved ones. The review copy
        # goes out in parallel; it does not block the asset.
        if run.routing.publishes and generate_assets:
            _notify(progress, "assets", "start")
            with tracer.start_as_current_span("assets") as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, TOOL_KIND)
                span.set_attribute("tool.name", "veo-3.1-lite-generate-preview")
                shot = S.build_video_prompt(
                    run.draft, client=client, sources=run.retrieval.sources_text
                )
                video_prompt, leaked, shot_request = shot.text, shot.leaked, shot.request
                run.cost.add(shot.cost)
                run.prompts.append({"step": "assets (shot brief)",
                                    "model": S.GENERATE_MODEL, "text": shot_request})
                run.prompts.append({"step": "assets (sent to Veo)",
                                    "model": "veo-3.1-lite-generate-preview",
                                    "text": video_prompt})
                # Visible in the trace: concept words that survived translation and
                # forced the fallback shot. A video model renders them literally.
                span.set_attribute("video.concept_terms_leaked", ",".join(leaked))
                span.set_attribute("video.used_fallback_prompt", bool(leaked))
                run.video_concept_leaked = list(leaked)
                span.set_attribute("video.has_turn", S.has_turn(video_prompt))
                # The video guardrail decides before a frame is generated.
                span.set_attribute("video.guardrail_layer",
                                   (shot.blocked.layer if shot.blocked else "none"))
                span.set_attribute("video.judge_verdict", shot.judge_label or "not_called")
                run.video_judge_label = shot.judge_label
                if shot.blocked is not None:
                    span.set_attribute("video.blocked", True)
                    span.set_attribute("video.block_type", shot.blocked.failure_type or "")
                    span.set_attribute("video.block_reason", shot.blocked.reason)
                    run.assets = S.AssetResult(video_prompt=video_prompt,
                                               error=shot.blocked.reason)
                    run.video_blocked = shot.blocked
                else:
                    span.set_attribute("video.blocked", False)
                    run.assets = S.generate_video(video_prompt)
                span.set_attribute(SpanAttributes.INPUT_VALUE, video_prompt)
                span.set_attribute(
                    "tool.status", "error" if run.assets.error else "success"
                )
                if run.assets.error:
                    span.set_attribute("tool.error", run.assets.error)
                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE,
                    run.assets.video_url or ("video_bytes" if run.assets.video_bytes else "none"),
                )
                _set_step_cost(span, shot.cost)
            _notify(progress, "assets", "done", {"error": run.assets.error, "has_video": bool(run.assets.video_bytes or run.assets.video_url)})

        # ── AUDIENCE ─────────────────────────────────────────────────────────
        # Only what actually published gets a reaction. A flagged campaign was
        # never seen, so nobody comments on it and it feeds nothing forward.
        if simulate_audience and run.routing.publishes:
            _notify(progress, "audience", "start")
            with tracer.start_as_current_span("audience-comments") as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, LLM_KIND)
                comments, aud_cost, aud_prompt = S.generate_audience_comments(
                    run.draft, cycle=cycle, client=client
                )
                run.audience_comments_generated = comments
                run.cost.add(aud_cost)
                run.prompts.append({"step": "audience", "model": S.GENERATE_MODEL,
                                    "text": aud_prompt})
                _set_llm_span(span, S.GENERATE_MODEL, aud_prompt,
                              json.dumps(comments), aud_cost)
                span.set_attribute("audience.n_comments", len(comments))
                span.set_attribute("audience.cycle", cycle)
            _notify(progress, "audience", "done", {"n": len(comments)})

        # ── Root attributes ──────────────────────────────────────────────────
        # The routing decision is made before ASSETS runs, so a blocked video is
        # not reflected in it. Surfaced separately rather than rewriting the
        # routing outcome — the copy genuinely did publish.
        root.set_attribute("trust.final_status", run.final_status)
        root.set_attribute("route.decision_before_assets", run.routing.final_status)
        root.set_attribute("trust.video_blocked", run.video_blocked is not None)
        root.set_attribute("trust.needs_human_review",
                           (not run.published) or run.routing.sampled)
        root.set_attribute("trust.check_layer", run.check_layer)
        root.set_attribute("trust.failure_type", run.failure_type or "")
        root.set_attribute("trust.claim_flagged", run.flagged_claim or "")
        root.set_attribute("trust.source_line", run.source_line or "")
        root.set_attribute("trust.judge_called",
                           any(e["judge_called"] for e in run.verify_events))
        root.set_attribute("trust.repair_attempted", run.repair_attempted)
        root.set_attribute("trust.repair_fixed", run.repair_fixed)
        # Emitted alongside the new names for one release so dashboards built on
        # the agent-era attributes keep working.
        root.set_attribute("trust.rewrite_attempted", run.repair_attempted)
        root.set_attribute("trust.rewrite_fixed", run.repair_fixed)
        root.set_attribute("trust.attempts", len(run.verify_events))
        root.set_attribute("trust.pipeline_halted", not run.published)
        root.set_attribute("trust.halt_reason", run.failure_type or "")
        root.set_attribute("trust.grounding_score_inherited", run.retrieval.grounding_score)
        root.set_attribute(
            "trust.retrieval_below_threshold",
            run.retrieval.grounding_score < RETRIEVAL_QUALITY_THRESHOLD,
        )

        root.set_attribute("cost.total_tokens", run.cost.total_tokens)
        root.set_attribute("cost.pricing_tier", run.cost.pricing_tier)
        if run.cost.total_usd is not None:
            root.set_attribute("cost.usd", run.cost.total_usd)
        else:
            root.set_attribute("cost.usd_available", False)
        if run.cost.tier_exceeded:
            root.set_attribute("cost.tier_exceeded", True)
        if run.cost.unmeasured_steps:
            root.set_attribute("cost.unmeasured_steps", ",".join(run.cost.unmeasured_steps))
        root.set_attribute("cost.breakdown", json.dumps(run.cost.breakdown()))

        root.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            f"{run.routing.final_status} | {run.draft.tagline}",
        )

        ctx = root.get_span_context()
        run.span_id = format(ctx.span_id, "016x") if ctx else None

    return run


def _event(attempt: int, verdict) -> dict:
    return {
        "attempt": attempt,
        "passed": verdict.passed,
        "layer": verdict.layer if not verdict.passed else "none",
        "failure_type": verdict.failure_type or "",
        "claim_flagged": verdict.claim_flagged or "",
        "source_line": verdict.source_line or "",
        "judge_called": verdict.judge_called,
    }
