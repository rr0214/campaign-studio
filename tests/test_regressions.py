"""
Regression tests for two defects found in review.

    python3 -m tests.test_regressions

1. brand_policy approved on a bare substring match against short keys, so any
   sentence containing "recycled" came back APPROVED — including
   "100% recycled materials", which contradicts its own source. The policy layer
   was stamping claims the documents refute.

2. price_usage returned 0.0 for a call whose usage could not be captured, making
   an unmeasured step indistinguishable from a genuinely free one. A cost of zero
   and a cost we failed to measure are different facts and must not collapse.
"""

import sys

import brand_policy
from brand_policy import check_brand_policy
from pipeline.pricing import CostMeter, StepCost, price_usage

FAILS = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILS.append(name)


# ---------------------------------------------------------------------------
print("\n=== 1. brand_policy: substring match approved contradicting claims ===")
# ---------------------------------------------------------------------------

# Overstatements — the approved claim is partial, so a universal quantifier
# applied to it is a different, larger claim.
for text in [
    "100% recycled materials",
    "fully recycled range",
    "every factory delivers performance",
    "all our materials are recycled",
    "entirely recycled fabric",
    "our performance is completely unmatched",
]:
    status = check_brand_policy(text)["status"]
    check(f"overstated: {text!r}", status != "APPROVED", f"got {status}")

# Negations — the approved wording is present but the sentence denies it.
for text in [
    "our materials are not recycled",
    "we don't use recycled polyester",
    "this range isn't recycled",
    "no recycled content here",
]:
    status = check_brand_policy(text)["status"]
    check(f"negated: {text!r}", status != "APPROVED", f"got {status}")

# The genuine article must still pass, or the fix has broken the feature.
for text in [
    "we use recycled polyester",
    "87% of materials by weight are certified sustainable",
    "45,000 garments collected since 2021",
    "performance-grade activewear",
]:
    status = check_brand_policy(text)["status"]
    check(f"still approved: {text!r}", status == "APPROVED", f"got {status}")

# Prohibited still wins, and is checked before anything else.
check("prohibited still prohibited",
      check_brand_policy("we are carbon neutral")["status"] == "PROHIBITED")

# A word that merely contains a key is not the key.
check("no false match inside a longer word",
      check_brand_policy("unrecycledness aside, we ship fast")["status"] != "APPROVED")

# An untraceable entry is never approved, even when phrased cleanly.
brand_policy.APPROVED_CLAIMS["packaging"] = brand_policy.ApprovedClaim(
    text="Recycled packaging (verified)",
    source_quote="We use recycled materials in our packaging",
    source_doc="verdant_sustainability.txt",
)
brand_policy._traceable_cache = None
check("untraceable entry not approved",
      check_brand_policy("our packaging is compostable")["status"] == "UNVERIFIED")
del brand_policy.APPROVED_CLAIMS["packaging"]
brand_policy._traceable_cache = None


# ---------------------------------------------------------------------------
print("\n=== 2. pricing: unmeasured must be unknown, not zero ===")
# ---------------------------------------------------------------------------

unmeasured = price_usage("verify", "gpt-5.6-terra", 0, 0, measured=False)
free = StepCost(step="verify", model="none (deterministic)", usd=0.0)
known = price_usage("generate", "gpt-5.6-luna", 1000, 200)

check("unmeasured usd is None", unmeasured.usd is None, repr(unmeasured.usd))
check("unmeasured state is 'unknown'", unmeasured.cost_state == "unknown", unmeasured.cost_state)
check("unmeasured displays as 'unknown'", unmeasured.display_usd() == "unknown",
      unmeasured.display_usd())

check("genuine zero usd is 0.0", free.usd == 0.0)
check("genuine zero state is 'free'", free.cost_state == "free", free.cost_state)
check("free and unknown are distinguishable", free.cost_state != unmeasured.cost_state)

check("known state is 'known'", known.cost_state == "known", known.cost_state)

# Unmeasured with non-zero tokens must still be unknown — the tokens were seen
# but the call was not, so pricing them would be a guess.
partial = price_usage("verify", "gpt-5.6-terra", 900, 100, measured=False)
check("unmeasured-with-tokens is still unknown", partial.usd is None, repr(partial.usd))
check("unmeasured keeps its token count", partial.total_tokens == 1000, str(partial.total_tokens))

# A total containing an unknown step must be withheld, not silently short.
m_unknown = CostMeter(); m_unknown.add(unmeasured); m_unknown.add(known)
check("total with an unknown step is None", m_unknown.total_usd is None,
      repr(m_unknown.total_usd))
check("total names what was not measured", "not measured" in m_unknown.display_total(),
      m_unknown.display_total())
check("has_unknown is true", m_unknown.has_unknown)

# A genuinely free step must NOT poison the total.
m_free = CostMeter(); m_free.add(free); m_free.add(known)
check("total with a free step still computes", m_free.total_usd == known.usd,
      repr(m_free.total_usd))
check("has_unknown is false when only free", not m_free.has_unknown)

# The breakdown carries the state so the UI cannot re-collapse them.
states = {k: v["state"] for k, v in m_unknown.breakdown().items()}
check("breakdown exposes per-step state", states.get("verify") == "unknown", str(states))

# An unpriceable model is also unknown, not free.
no_price = price_usage("x", "some-model-we-have-no-rate-for", 1000, 100)
check("unknown model priced as unknown", no_price.usd is None, repr(no_price.usd))


# ---------------------------------------------------------------------------
print("\n=== 3. a failed judge call costs 'unknown', and the eval mean skips it ===")
# ---------------------------------------------------------------------------
# Fail-closed already covers the verdict; this covers the accounting. A judge that
# errored still consumed no measurable usage, so its cost must be unknown — and an
# unknown must not be averaged in as zero, which would quietly drag the mean down.
from unittest import mock

import pandas as pd

import evals.grounding_check as gc
import pipeline.steps as S

_draft = S.CampaignDraft(tagline="T", campaign_concept="87% certified.", caption="c")
_retrieval = type("R", (), {"sources_text": "87% of materials by weight are certified sustainable"})()

with mock.patch.object(gc, "run_judge", side_effect=RuntimeError("judge exploded")):
    verdict = S.verify(_draft, _retrieval)

check("failed judge fails closed", not verdict.passed, str(verdict.failure_type))
check("failed judge cost is unknown", verdict.cost.cost_state == "unknown",
      f"{verdict.cost.cost_state} / usd={verdict.cost.usd!r}")
check("failed judge is not priced as free", verdict.cost.usd is None, repr(verdict.cost.usd))

_known = price_usage("generate", "gpt-5.6-luna", 1000, 200)
_meter = CostMeter(); _meter.add(verdict.cost); _meter.add(_known)
check("run total withheld after a judge failure", _meter.total_usd is None,
      repr(_meter.total_usd))

# Call the eval's own aggregation rather than replicating it — a replica would
# keep passing after the real one changed.
from evals.eval_end_to_end import summarise_costs

_frame = pd.DataFrame({"usd": [_meter.total_usd, 0.004, 0.006]})
_costs = summarise_costs(_frame)
check("eval counts the unknown run", _costs["n_unknown"] == 1, str(_costs))
check("eval prices only the measured runs", _costs["n_priced"] == 2, str(_costs))
check("eval mean excludes the unknown", abs(_costs["mean_usd"] - 0.005) < 1e-9,
      f"mean={_costs['mean_usd']} (0.0033 would mean it counted unknown as zero)")
check("eval total excludes the unknown", abs(_costs["total_usd"] - 0.010) < 1e-9,
      f"total={_costs['total_usd']}")

# All-unknown must not report $0.00 across the board.
_all_unknown = summarise_costs(pd.DataFrame({"usd": [None, None]}))
check("all-unknown reports no total", _all_unknown["total_usd"] is None, str(_all_unknown))
check("all-unknown reports no mean", _all_unknown["mean_usd"] is None, str(_all_unknown))


print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("All regression checks passed.")
