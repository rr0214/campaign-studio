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


# ---------------------------------------------------------------------------
print("\n=== 4. video guardrail layer 1 blocks generation, fail-closed ===")
# ---------------------------------------------------------------------------
# A shot description goes to a video model that renders what it is told. If a
# prohibited claim survives into the shot, the campaign guardrail never sees it —
# it only reads copy. Layer 1 runs the same deterministic checks on the shot.
import pipeline.steps as _S

_blocked = _S._guard_shot(
    "A runner crests a ridge beneath a banner reading carbon neutral since 2025. "
    "Camera low, tracking, dawn light.",
    price_usage("assets.prompt", "gpt-5.6-luna", 100, 50), [], "", None, None,
)
check("carbon-neutral shot is blocked", not _blocked.safe, str(_blocked.blocked))
check("blocked shot names the failure type",
      _blocked.blocked.failure_type == "PROHIBITED_PHRASE",
      str(_blocked.blocked.failure_type))
check("blocked shot names the phrase", "carbon neutral" in _blocked.blocked.claim_flagged,
      _blocked.blocked.claim_flagged)

_clean = _S._guard_shot(
    "Two hands stretch a knit panel taut and let go; it snaps flat. Camera at hand "
    "height, shallow focus, low side light.",
    price_usage("assets.prompt", "gpt-5.6-luna", 100, 50), [], "", None, None,
)
check("clean shot passes layer 1", _clean.safe, str(_clean.blocked))

# An unapproved statistic in a shot is blocked the same way.
_stat = _S._guard_shot(
    "A wall of 400 reclaimed bottles fills the frame, then a hand lifts one away. "
    "Camera static, daylight.",
    price_usage("assets.prompt", "gpt-5.6-luna", 100, 50), [], "", None, None,
)
check("unapproved stat in a shot is blocked", not _stat.safe,
      str(_stat.blocked.failure_type if _stat.blocked else None))

# Layer 2 fails closed when the judge cannot run.
from unittest import mock as _mock
import evals.grounding_check as _gc
with _mock.patch.object(_gc, "judge_rows", side_effect=RuntimeError("judge down")):
    _nojudge = _S._guard_shot(
        "Two hands stretch a knit panel taut and let go; it snaps flat.",
        price_usage("assets.prompt", "gpt-5.6-luna", 100, 50), [], "",
        "87% of materials by weight are certified sustainable", None,
    )
check("video judge unavailable fails closed", not _nojudge.safe,
      str(_nojudge.blocked.failure_type if _nojudge.blocked else None))
check("judge-unavailable is typed as such",
      _nojudge.blocked.failure_type == "JUDGE_UNAVAILABLE",
      str(_nojudge.blocked.failure_type))


# ---------------------------------------------------------------------------
print("\n=== 5. the series loop actually closes ===")
# ---------------------------------------------------------------------------
# Audience comments are generated, stored and passed forward through three
# separate hops. Nothing asserted that they arrive. A break anywhere in that
# chain would be silent: cycle 2 would simply write a campaign that ignores
# what the audience said, and look perfectly fine doing it.
import json as _json
from unittest import mock as _m2

from pipeline.steps import _build_generate_prompt
from pipeline.run import run_campaign_pipeline
from pipeline.retrieval import RetrievalResult

_COMMENTS = [
    "I thought you were already carbon neutral honestly",
    "87% certified sustainable is impressive, keep leading on this",
]
_CORRECTION = {"cycle": 1, "claim": "100% recycled",
               "source_line": "87% of materials by weight are certified sustainable",
               "correction": "87% of materials by weight", "resolved": True}

# a) the prompt builder puts them in
_prompt = _build_generate_prompt(
    brief="Spring materials story", sources_text="87% of materials by weight...",
    audience_comments=_COMMENTS,
    previous_copy={"tagline": "Move With Purpose", "key_messages": ["87% certified"]},
    cycle=2, prior_corrections=[_CORRECTION],
)
for c in _COMMENTS:
    check(f"comment reaches the brief: {c[:38]!r}...", c in _prompt)
check("previous approved copy reaches the brief", "Move With Purpose" in _prompt)
check("prior correction reaches the brief", _CORRECTION["claim"] in _prompt)
check("the source line behind it reaches the brief", _CORRECTION["source_line"] in _prompt)

# b) the pipeline threads them all the way into the captured prompt
_ret = RetrievalResult(query="q",
    chunks=[{"text": "87% of materials by weight are certified sustainable",
             "doc_name": "verdant_sustainability.txt", "relevance_score": 0.9}],
    sources=["verdant_sustainability.txt"], grounding_score=0.9,
    metadata={"retrieval_quality": "high", "injection_risk": "low"})

_payload = _json.dumps({"tagline": "T", "campaign_concept": "87% by weight.",
                        "key_messages": ["m"], "caption": "c", "hashtags": ["#v"],
                        "claims": []})
class _Usage:
    prompt_tokens = 10; completion_tokens = 5
    completion_tokens_details = type("D", (), {"reasoning_tokens": 0})()
    prompt_tokens_details = type("D", (), {"cached_tokens": 0})()
class _Comp:
    def create(self, **kw):
        msg = type("M", (), {"content": _payload})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()],
                              "usage": _Usage()})()
_client = type("Cl", (), {"chat": type("Ch", (), {"completions": _Comp()})()})()

with _m2.patch("pipeline.run.retrieve", lambda *a, **k: _ret), \
     _m2.patch.object(_gc, "run_judge", lambda o, s, model=None: ("supported", "ok")):
    _run2 = run_campaign_pipeline(
        brief="Spring materials story", cycle=2, sample_rate=0.0,
        generate_assets=False, client=_client,
        audience_comments=_COMMENTS, prior_corrections=[_CORRECTION],
        previous_copy={"tagline": "Move With Purpose", "key_messages": ["87% certified"]},
    )
_gen = next((p["text"] for p in _run2.prompts if p["step"] == "generate"), "")
check("cycle-2 generate prompt was captured", bool(_gen), f"{len(_gen)} chars")
for c in _COMMENTS:
    check(f"comment survives the pipeline: {c[:34]!r}...", c in _gen)
check("correction survives the pipeline", _CORRECTION["source_line"] in _gen)


# ---------------------------------------------------------------------------
print("\n=== 6. a blocked video holds the WHOLE campaign ===")
# ---------------------------------------------------------------------------
# ROUTE decides before ASSETS runs, so routing alone cannot know the shot was
# refused. A campaign is its copy and its video together: if any check fails on
# any surface, nothing publishes. Partial publication is not an outcome.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("appmod_reg", "app.py")
_app = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_app)

from evals.grounding_check import GroundingCheckResult as _GCR
from pipeline.run import CampaignRun as _CR
from pipeline.steps import RouteResult as _RR


def _run_with(video_blocked=None, sampled=False, n=3):
    """A real CampaignRun, so published/final_status/halt_reason are the real ones."""
    r = _CR(brief="b", cycle=1)
    r.routing = _RR(destination="human_sample" if sampled else "publish",
                    sampled=sampled,
                    final_status="SAMPLED" if sampled else "APPROVED",
                    publishes=True,
                    receipts=[{"source_found": True}] * n)
    r.video_blocked = video_blocked
    return r


_block = _GCR(passed=False, layer="judge", failure_type="UNSUPPORTED_CLAIM",
              claim_flagged="A viewer would conclude the factory runs entirely on solar.",
              reason="Video blocked: a viewer would conclude something the brand "
                     "documents do not support.",
              judge_called=True, judge_label="unsupported")

# behaviour, not just wording
_clean = _run_with()
check("clean run still publishes", _clean.published)
check("clean run status APPROVED", _clean.final_status == "APPROVED", _clean.final_status)
check("clean run has no halt reason", _clean.halt_reason is None)

_held = _run_with(video_blocked=_block)
check("blocked video means NOT published", not _held.published)
check("blocked video status is FLAGGED", _held.final_status == "FLAGGED", _held.final_status)
check("blocked video surfaces the video reason",
      "Video blocked" in (_held.halt_reason or ""), (_held.halt_reason or "")[:50])

# the chip
_cls, _head, _sub = _app.status_state(_held)
check("chip says held for review", _head == "Held for review — video blocked", _head)
check("chip no longer claims partial publication",
      "Published" not in _head and "copy only" not in _head, _head)
check("chip uses the review class", _cls == "s-review", _cls)
check("chip subtext carries the reason", "do not support" in _sub, _sub[:60])

_cls2, _head2, _ = _app.status_state(_run_with())
check("clean chip unchanged", _head2 == "Published — no review needed", _head2)

# a sample selection does not rescue a blocked video
_cls3, _head3, _ = _app.status_state(_run_with(video_blocked=_block, sampled=True))
check("blocked video outranks the audit sample",
      _head3 == "Held for review — video blocked", _head3)
check("sampled+blocked still does not publish",
      not _run_with(video_blocked=_block, sampled=True).published)

# a deterministic block behaves identically
_det = _GCR(passed=False, layer="deterministic", failure_type="PROHIBITED_PHRASE",
            claim_flagged="carbon neutral",
            reason='Video blocked: the shot description contains the prohibited claim.')
check("deterministic video block also holds the campaign",
      not _run_with(video_blocked=_det).published)


# ---------------------------------------------------------------------------
print("\n=== 7. a blocked video's reason must not describe a halted campaign ===")
# ---------------------------------------------------------------------------
# The same checks run over copy and over the shot description. The message used
# to be written for copy only, so a blocked video read "Pipeline halted:
# campaign text cites the statistic…" — wrong twice over. The pipeline did not
# halt (the copy published) and it was not campaign text.
from evals.grounding_check import (SUBJECT_VIDEO as _SV, SUBJECT_COPY as _SC,
                                   run_deterministic_checks)

for _text, _what in [("We are at 100% recycled.", "unapproved stat"),
                     ("Verdant is carbon neutral.", "prohibited phrase"),
                     ("Softer than Lululemon.", "competitor")]:
    _v = run_deterministic_checks(_text, subject=_SV)
    check(f"video reason avoids 'Pipeline halted' ({_what})",
          "Pipeline halted" not in _v.reason, _v.reason[:60])
    check(f"video reason avoids 'campaign text' ({_what})",
          "campaign text" not in _v.reason, _v.reason[:60])
    check(f"video reason names the shot ({_what})",
          "shot description" in _v.reason, _v.reason[:60])
    check(f"video reason says blocked, not halted ({_what})",
          _v.reason.startswith("Video blocked"), _v.reason[:40])

_c = run_deterministic_checks("We are at 100% recycled.", subject=_SC)
check("copy reason still names campaign text", "campaign text" in _c.reason, _c.reason[:60])
check("copy reason says campaign halted", _c.reason.startswith("Campaign halted"),
      _c.reason[:40])
check("copy default subject is copy",
      run_deterministic_checks("We are at 100% recycled.").reason == _c.reason)

# the checks themselves must be identical — only the wording differs
for _t in ["We are at 100% recycled.", "Verdant is carbon neutral.",
           "Softer than Lululemon.", "87% of materials by weight are certified sustainable."]:
    _a = run_deterministic_checks(_t, subject=_SC)
    _b = run_deterministic_checks(_t, subject=_SV)
    same = (_a is None and _b is None) or (
        _a is not None and _b is not None
        and _a.failure_type == _b.failure_type and _a.claim_flagged == _b.claim_flagged)
    check(f"same verdict either subject: {_t[:34]!r}", same)


# ---------------------------------------------------------------------------
print("\n=== 8. camera specs are not brand claims (shot path only) ===")
# ---------------------------------------------------------------------------
# "35mm lens" tripped UNAPPROVED_STAT on the shot description, so no video ever
# generated. Lens lengths, frame rates and apertures are cinematography. A bare
# number in a shot is still a claim, and the copy path is unchanged.
from evals.grounding_check import check_unapproved_stat as _cus

for _spec in ["Camera at 35mm, waist height.", "Shot at 24fps, 180 degree shutter.",
              "Wide open at f/2.8, handheld.", "A 50mm lens, 2x speed ramp.",
              "Holds for 5 seconds, then cuts.", "Tracking at 12 ft, low angle."]:
    _v = run_deterministic_checks(_spec, subject=_SV)
    check(f"spec does not block video: {_spec[:34]!r}", _v is None,
          str(_v.claim_flagged if _v else ""))

for _claim in ["A wall of 400 reclaimed bottles fills the frame.",
               # 99 would pass — it is an approved stat (99% solvent recovery).
               "A sign showing 250 factories fills the frame."]:
    _v = run_deterministic_checks(_claim, subject=_SV)
    check(f"bare number still blocks video: {_claim[:34]!r}", _v is not None,
          str(_v.failure_type if _v else "PASSED"))

# mixed: a spec and a claim in one shot — the claim must still be caught
_mixed = run_deterministic_checks(
    "Camera at 35mm, f/2.8, as a wall of 400 bottles fills the frame.", subject=_SV)
check("a claim beside camera specs is still caught",
      _mixed is not None and "400" in (_mixed.claim_flagged or ""),
      str(_mixed.claim_flagged if _mixed else "PASSED"))

# the copy path is untouched — a measurement in copy is still a claim
check("copy path still flags 35mm",
      run_deterministic_checks("Camera at 35mm, waist height.") is not None)
check("check_unapproved_stat defaults to flagging measurements",
      _cus("Camera at 35mm.") is not None)


# ---------------------------------------------------------------------------
print("\n=== 9. a video block is recorded as the failure ===")
# ---------------------------------------------------------------------------
# The audit record printed failure_type None beside status FLAGGED, because only
# the copy verdict was consulted and the copy had passed.
_vblock = _GCR(passed=False, layer="judge", failure_type="UNSUPPORTED_CLAIM",
               claim_flagged="viewer would conclude full solar",
               reason="Video blocked: a viewer would conclude something unsupported.",
               judge_called=True, judge_label="unsupported")
_r = _run_with(video_blocked=_vblock)
check("failure_type reports the video failure",
      _r.failure_type == "UNSUPPORTED_CLAIM", str(_r.failure_type))
check("failure_type is not None on a FLAGGED run",
      _r.failure_type is not None and _r.final_status == "FLAGGED",
      f"{_r.failure_type} / {_r.final_status}")
check("check_layer reports the layer that blocked", _r.check_layer == "judge",
      _r.check_layer)

# a video judge outage must still read as a video block, not a copy-judge outage
_vout = _GCR(passed=False, layer="judge", failure_type="JUDGE_UNAVAILABLE",
             claim_flagged="", reason="Video blocked: the visual-claim judge could not run.",
             judge_called=True)
_cls_o, _head_o, _ = _app.status_state(_run_with(video_blocked=_vout))
check("video judge outage reads as a video block",
      _head_o == "Held for review — video blocked", _head_o)


print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("All regression checks passed.")
