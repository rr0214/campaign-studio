"""
Pipeline tests with mocked model calls. No network, no spend.

Run:  python3 -m tests.test_pipeline
"""

import json
import random
import sys
from unittest import mock

import evals.grounding_check as gc
import pipeline.steps as S
from pipeline.run import run_campaign_pipeline
from pipeline.retrieval import RetrievalResult

SOURCES = [
    {"text": "87% of materials by weight are certified sustainable (up from 81% in 2024).",
     "doc_name": "verdant_sustainability.txt", "relevance_score": 0.9},
    {"text": "Primary Factory: Marisol, Portugal\n- Fair Trade USA certified since 2021\n"
             "Secondary Factory: Gaia Textiles, Sri Lanka\n"
             "- Do NOT cite this factory as Fair Trade certified in any materials",
     "doc_name": "verdant_sustainability.txt", "relevance_score": 0.85},
]


def fake_retrieval():
    return RetrievalResult(
        query="q", chunks=SOURCES, sources=["verdant_sustainability.txt"],
        grounding_score=0.88,
        metadata={"n_chunks_retrieved": 2, "avg_relevance": 0.87,
                  "retrieval_quality": "high", "injection_risk": "low",
                  "simulated_low_confidence": False},
    )


class FakeUsage:
    prompt_tokens = 1200
    completion_tokens = 400
    completion_tokens_details = type("D", (), {"reasoning_tokens": 150})()
    prompt_tokens_details = type("D", (), {"cached_tokens": 200})()


def fake_client(payloads):
    """Returns a client whose successive create() calls yield `payloads` in order."""
    seq = list(payloads)

    class Completions:
        def create(self, **kw):
            content = seq.pop(0) if seq else "{}"
            msg = type("M", (), {"content": content})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()],
                                  "usage": FakeUsage()})()

    return type("Client", (), {"chat": type("Ch", (), {"completions": Completions()})()})()


def campaign(tagline, concept, claim_text="87% of materials by weight are certified sustainable"):
    return json.dumps({
        "tagline": tagline, "campaign_concept": concept,
        "key_messages": ["Built for movement."], "caption": "Move with purpose.",
        "hashtags": ["#Verdant"],
        "claims": [{"text": claim_text, "kind": "statistic",
                    "source_quote": "87% of materials by weight are certified sustainable"}],
    })


def repaired(tagline, concept, corrected):
    d = json.loads(campaign(tagline, concept))
    d["corrected_claim"] = corrected
    return json.dumps(d)


FAILS = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILS.append(name)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def run(name, payloads, judge, sample_rate=0.0, rng=None):
    print(f"\n=== {name} ===")
    with mock.patch("pipeline.run.retrieve", lambda *a, **k: fake_retrieval()), \
         mock.patch.object(gc, "run_judge", judge):
        return run_campaign_pipeline(
            brief="Spring materials story",
            client=fake_client(payloads),
            sample_rate=sample_rate,
            rng=rng,
            generate_assets=False,
        )


# ── 1. clean campaign passes on the first check ─────────────────────────────
r = run("clean → APPROVED",
        [campaign("Move With Purpose", "87% of materials by weight are certified sustainable.")],
        lambda o, s, model=None: ("supported", "traces to sources"))
check("final_status APPROVED", r.final_status == "APPROVED", r.final_status)
check("publishes", r.published)
check("no repair", not r.repair_attempted)
check("judge was called", r.verify_events[0]["judge_called"])
check("one verify attempt", len(r.verify_events) == 1)
check("receipts attached", len(r.routing.receipts) == 1)
check("receipt located a source", r.routing.receipts[0]["source_found"])
check("cost has generate step", "generate" in r.cost.breakdown())
# Under mocks the judge never makes a real API call, so its usage is genuinely
# uncaptured and the run total is correctly withheld. It used to come back as
# $0.00 — see tests/test_regressions.py, defect 2. What must still hold is that
# the steps we DO own are priced.
check("generate step is priced", r.cost.breakdown()["generate"]["state"] == "known",
      str(r.cost.breakdown()["generate"]))
check("total withheld while a step is unmeasured", r.cost.total_usd is None,
      repr(r.cost.total_usd))

# ── 2. deterministic catch, judge never called, repair fixes it ─────────────
judge_calls = {"n": 0}


def counting_judge(o, s, model=None):
    judge_calls["n"] += 1
    return ("supported", "fine")


r = run("unapproved stat → repair → APPROVED",
        [campaign("Ninety Nine", "100% of materials by weight are certified sustainable."),
         repaired("Move With Purpose", "87% of materials by weight are certified sustainable.",
                  "87% of materials by weight are certified sustainable")],
        counting_judge)
check("first check caught by deterministic", r.verify_events[0]["layer"] == "deterministic",
      r.verify_events[0]["layer"])
check("failure_type UNAPPROVED_STAT", r.failure_type == "UNAPPROVED_STAT", str(r.failure_type))
check("judge NOT called on attempt 1", not r.verify_events[0]["judge_called"])
check("repair attempted", r.repair_attempted)
check("repair fixed it", r.repair_fixed)
check("final APPROVED", r.final_status == "APPROVED", r.final_status)
check("source line located", bool(r.source_line), str(r.source_line)[:50])
check("corrected claim recorded", bool(r.corrected_claim), r.corrected_claim)
check("exactly one judge call total", judge_calls["n"] == 1, str(judge_calls["n"]))

# ── 3. repair fails → FLAGGED, no publish ───────────────────────────────────
r = run("repair fails → FLAGGED",
        [campaign("Ninety Nine", "100% of our fabric is recycled."),
         repaired("Still Wrong", "77% of our fabric is recycled.", "77% recycled")],
        lambda o, s, model=None: ("supported", "fine"))
check("final FLAGGED", r.final_status == "FLAGGED", r.final_status)
check("does not publish", not r.published)
check("destination human_flagged", r.routing.destination == "human_flagged")
check("two verify attempts", len(r.verify_events) == 2, str(len(r.verify_events)))
check("original draft kept", "100%" in r.original_draft.verified_text())
check("repaired draft differs", "77%" in r.draft.verified_text())
check("halt_reason names the claim", "77%" in (r.halt_reason or ""), (r.halt_reason or "")[:70])
check("no receipts when flagged", r.routing.receipts == [])

# ── 4. judge flags scope broadening ─────────────────────────────────────────
r = run("judge flags broadened claim → FLAGGED",
        [campaign("Ethically Made", "Every factory we work with is Fair Trade USA certified.",
                  claim_text="Every factory is Fair Trade USA certified"),
         repaired("Still Broad", "All our factories are Fair Trade USA certified.", "all factories")],
        lambda o, s, model=None: ("unsupported", "Fair Trade claim covers all manufacturing; sources cover Portugal only"))
check("caught by judge", r.check_layer == "judge", r.check_layer)
check("failure UNSUPPORTED_CLAIM", r.failure_type == "UNSUPPORTED_CLAIM", str(r.failure_type))
check("final FLAGGED", r.final_status == "FLAGGED", r.final_status)
check("judge cost reported as unknown under mocks",
      r.cost.breakdown().get("verify", {}).get("state") == "unknown",
      str(r.cost.breakdown().get("verify")))

# ── 5. sampling: sampled items still publish ────────────────────────────────
always = random.Random()
always.random = lambda: 0.0          # below any positive rate
never = random.Random()
never.random = lambda: 0.99

r = run("sampled → SAMPLED and still publishes",
        [campaign("Move With Purpose", "87% of materials by weight are certified sustainable.")],
        lambda o, s, model=None: ("supported", "fine"), sample_rate=0.10, rng=always)
check("final SAMPLED", r.final_status == "SAMPLED", r.final_status)
check("sampled flag true", r.routing.sampled)
check("destination human_sample", r.routing.destination == "human_sample")
check("STILL PUBLISHES (the override)", r.published)
check("receipts still attached", len(r.routing.receipts) == 1)

r = run("not sampled → APPROVED",
        [campaign("Move With Purpose", "87% of materials by weight are certified sustainable.")],
        lambda o, s, model=None: ("supported", "fine"), sample_rate=0.10, rng=never)
check("final APPROVED", r.final_status == "APPROVED", r.final_status)
check("not sampled", not r.routing.sampled)
check("publishes", r.published)

# ── 6. fail-closed when the judge is unavailable ────────────────────────────
def exploding_judge(o, s, model=None):
    raise RuntimeError("phoenix unavailable")


r = run("judge unavailable → fail closed",
        [campaign("Move With Purpose", "87% of materials by weight are certified sustainable."),
         repaired("Same", "87% of materials by weight are certified sustainable.", "unchanged")],
        exploding_judge)
check("final FLAGGED (not published)", r.final_status == "FLAGGED", r.final_status)
check("failure JUDGE_UNAVAILABLE", r.failure_type == "JUDGE_UNAVAILABLE", str(r.failure_type))

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("All pipeline checks passed.")
