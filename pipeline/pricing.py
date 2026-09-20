"""
Token accounting and USD pricing.
=================================

Tokens are measured; USD is computed from the table below. Nothing here
estimates — if a price is missing the cost is reported as unavailable rather
than guessed, and if a call's context would cross into the long-context tier the
run is flagged rather than priced at the cheaper rate.
"""

from dataclasses import dataclass, field
from typing import Optional


# USD per 1M tokens, standard tier, short context.
PRICES_USD_PER_1M = {
    "gpt-5.6-luna":  {"input": 0.10, "cached_input": 0.01, "output": 0.60},
    "gpt-5.6-terra": {"input": 1.00, "cached_input": 0.10, "output": 6.00},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
}

# The long-context tier is priced roughly 2x the rates above. This is the prompt
# size at which it is assumed to engage.
#
# ASSUMPTION, NOT VERIFIED: the boundary below is the conventional 128k figure.
# The exact threshold for the gpt-5.6 family was not confirmed against a price
# sheet. It is set conservatively low on purpose — crossing it produces a loud
# flag, never a silently cheaper number. Verdant campaigns run 2-5k tokens, so
# this should never engage; if it does, the pricing needs checking before the
# figure is trusted.
LONG_CONTEXT_THRESHOLD_TOKENS = 128_000
LONG_CONTEXT_MULTIPLIER = 2.0

TIER_SHORT = "standard/short-context"
TIER_LONG = "standard/long-context"
TIER_UNKNOWN = "unknown"


@dataclass
class StepCost:
    """Token usage and cost for one pipeline step."""
    step: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    usd: Optional[float] = None
    pricing_tier: str = TIER_SHORT
    measured: bool = True          # False when usage could not be captured
    tier_exceeded: bool = False    # True when the prompt crossed the long-context boundary

    @property
    def usd_available(self) -> bool:
        return self.usd is not None


def price_usage(
    step: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    measured: bool = True,
) -> StepCost:
    """
    Price one call. Reasoning tokens are already included in completion_tokens by
    the API and bill as output, so they are recorded but not added again. Cached
    input tokens are a subset of prompt_tokens and bill at the cached rate, so
    they are subtracted out of the full-price portion.
    """
    total = prompt_tokens + completion_tokens

    tier_exceeded = prompt_tokens > LONG_CONTEXT_THRESHOLD_TOKENS
    tier = TIER_LONG if tier_exceeded else TIER_SHORT

    rates = PRICES_USD_PER_1M.get(model)
    if rates is None or any(rates.get(k) is None for k in ("input", "cached_input", "output")):
        return StepCost(
            step=step, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens,
            total_tokens=total, usd=None, pricing_tier=TIER_UNKNOWN,
            measured=measured, tier_exceeded=tier_exceeded,
        )

    multiplier = LONG_CONTEXT_MULTIPLIER if tier_exceeded else 1.0
    full_price_input = max(prompt_tokens - cached_tokens, 0)

    usd = (
        full_price_input * rates["input"]
        + cached_tokens * rates["cached_input"]
        + completion_tokens * rates["output"]
    ) / 1_000_000 * multiplier

    return StepCost(
        step=step, model=model,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens,
        total_tokens=total, usd=round(usd, 8), pricing_tier=tier,
        measured=measured, tier_exceeded=tier_exceeded,
    )


def cost_from_response(step: str, model: str, response) -> StepCost:
    """Build a StepCost from an OpenAI SDK response's `usage` block."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return StepCost(step=step, model=model, measured=False, pricing_tier=TIER_UNKNOWN)

    det_out = getattr(usage, "completion_tokens_details", None)
    det_in = getattr(usage, "prompt_tokens_details", None)
    return price_usage(
        step=step,
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        reasoning_tokens=(getattr(det_out, "reasoning_tokens", 0) or 0) if det_out else 0,
        cached_tokens=(getattr(det_in, "cached_tokens", 0) or 0) if det_in else 0,
    )


@dataclass
class CostMeter:
    """Accumulates per-step costs across one campaign run."""
    steps: list = field(default_factory=list)

    def add(self, step_cost: StepCost) -> StepCost:
        self.steps.append(step_cost)
        return step_cost

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def total_usd(self) -> Optional[float]:
        """None if any measured step could not be priced — a partial total would mislead."""
        if not self.steps:
            return 0.0
        if any(s.usd is None for s in self.steps):
            return None
        return round(sum(s.usd for s in self.steps), 8)

    @property
    def pricing_tier(self) -> str:
        if any(s.pricing_tier == TIER_UNKNOWN for s in self.steps):
            return TIER_UNKNOWN
        return TIER_LONG if any(s.tier_exceeded for s in self.steps) else TIER_SHORT

    @property
    def tier_exceeded(self) -> bool:
        return any(s.tier_exceeded for s in self.steps)

    @property
    def unmeasured_steps(self) -> list:
        return [s.step for s in self.steps if not s.measured]

    def breakdown(self) -> dict:
        return {s.step: {"tokens": s.total_tokens, "usd": s.usd} for s in self.steps}
