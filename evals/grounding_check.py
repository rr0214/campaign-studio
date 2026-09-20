"""
Claim Grounding Check
=====================
Checks whether generated campaign text is actually supported by the brand
documents the pipeline retrieved — as opposed to whether retrieval *found*
relevant-looking documents, which is all `_calculate_grounding_score()` in
Agent 1 measures.

The cascade, cheapest first:

  Layer 1 — deterministic (free, no model calls)
      1a. prohibited phrase   → reuses check_brand_policy() from Agent 2
      1b. unapproved statistic → any number not in the brand guide's approved set
      1c. competitor named     → brand guide prohibits naming competitors

  Layer 2 — LLM judge (only reached by text that passed layer 1)
      Compares the campaign text against the chunks Agent 1 actually
      retrieved. Comparing against retrieved chunks rather than all four
      brand docs is deliberate: it makes retrieval failure visible.

The failure this exists to catch is a dropped qualifier. The sources say
"our Portugal factory is Fair Trade certified"; the campaign says "our
manufacturing is Fair Trade certified". No prohibited phrase was added, so
check_brand_policy() passes it — but Verdant's second factory (Sri Lanka)
is not certified, so the claim is now false.

Strictness is tuned for RECALL. A missed unsupported claim reaches a human
who may publish it (legal exposure). A false alarm costs a reviewer thirty
seconds and saves a video-generation call.

FAIL-CLOSED: if the judge cannot run (phoenix missing, API error), the check
HALTS rather than waving the campaign through. Same asymmetry — an
unavailable judge is not evidence that the text is grounded.
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from brand_policy import check_brand_policy

# ---------------------------------------------------------------------------
# Failure types (span attribute values — keep stable, they're queried in Arize)
# ---------------------------------------------------------------------------

PROHIBITED_PHRASE = "PROHIBITED_PHRASE"
UNAPPROVED_STAT = "UNAPPROVED_STAT"
COMPETITOR_NAMED = "COMPETITOR_NAMED"
UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
JUDGE_UNAVAILABLE = "JUDGE_UNAVAILABLE"

LAYER_DETERMINISTIC = "deterministic"
LAYER_JUDGE = "judge"
LAYER_NONE = "none"


# ---------------------------------------------------------------------------
# 1b. Unapproved statistic
# ---------------------------------------------------------------------------

# Every numeric claim Verdant is cleared to make, normalized to a bare number.
# Source: brand guide "APPROVED STATISTICS" + sustainability report figures.
APPROVED_STATS = {
    87: "87% of materials certified sustainable",
    45000: "45,000+ garments in Take Back Program",
    45312: "45,312 garments collected (exact figure)",
    92: "92% customer satisfaction",
    3: "3-year product warranty",
    8: "8% sizing return rate / landfill share of Take Back",
    68: "68% solar share, Portugal factory",
    12: "12 plastic bottles diverted per legging",
    99: "99% solvent recovery (TENCEL closed loop)",
    61: "61% of Take Back garments recycled",
    31: "31% of Take Back garments donated",
}

# Matches 87, 87%, 45,000, 45,000+, 3.2, $98. The lookbehind skips digits glued
# to a letter ("Q2", "XS-3X"), which are labels rather than statistics.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\$?\d[\d,]*(?:\.\d+)?\+?%?")


def _normalize(token: str) -> Optional[float]:
    """'45,000+' -> 45000.0, '87%' -> 87.0. None if it isn't a number."""
    cleaned = token.strip("$%+").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_excluded(token: str, value: float) -> bool:
    """
    Two narrow exclusions, so the check fires on statistics rather than on
    every digit in the copy:
      - calendar years (2019, Q2 2026) — dates, not brand statistics
      - currency amounts ($98) — prices live in the product catalog
    Everything else, including product composition percentages, is compared
    against APPROVED_STATS and flagged if absent. That over-fires by design.
    """
    if token.startswith("$"):
        return True
    if not token.endswith("%") and value.is_integer() and 1900 <= value <= 2100:
        return True
    return False


def check_unapproved_stat(text: str) -> Optional[tuple[str, str]]:
    """Return (offending_number, surrounding_text) for the first unapproved stat."""
    for match in _NUMBER_RE.finditer(text or ""):
        token = match.group()
        value = _normalize(token)
        if value is None or _is_excluded(token, value):
            continue
        if value not in APPROVED_STATS:
            start = max(0, match.start() - 45)
            end = min(len(text), match.end() + 45)
            return token, text[start:end].strip()
    return None


# ---------------------------------------------------------------------------
# 1c. Competitor named
# ---------------------------------------------------------------------------

COMPETITORS = [
    "Lululemon", "Athleta", "Patagonia", "Nike", "Alo",
    "Vuori", "Girlfriend Collective", "Outdoor Voices",
]


def check_competitor_named(text: str) -> Optional[str]:
    """Return the first competitor name found, case-insensitive."""
    lowered = (text or "").lower()
    for competitor in COMPETITORS:
        if re.search(rf"\b{re.escape(competitor.lower())}\b", lowered):
            return competitor
    return None


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

# Version-controlled: the judge we validate has to be the judge that ships.
# Changing this prompt invalidates the TPR/TNR numbers in evals/judge_validation.csv.
GROUNDING_JUDGE_TEMPLATE = """You are checking whether a marketing campaign stays within what the source
documents actually support.

SOURCE DOCUMENTS (everything the system retrieved):
{sources}

CAMPAIGN TEXT TO CHECK:
{output}

Check only VERIFIABLE FACTUAL CLAIMS: numbers, percentages, certifications,
partnerships, materials, factory locations, program results, dates.
IGNORE marketing voice, tone, aspiration, and stylistic language — those are
not checkable and are not your concern.

A claim is UNSUPPORTED if any of these is true:
- It states something the source documents do not say
- It contradicts the source documents
- It BROADENS the scope of something the sources state narrowly.
  Example: sources say "our Portugal factory is Fair Trade certified";
  campaign says "our manufacturing is Fair Trade certified". The qualifier
  was dropped, so the claim is now broader than the source. UNSUPPORTED.
- It restates a prohibited claim in different words
  (e.g. "net zero", "climate positive", "we've balanced our footprint"
  all restate carbon neutrality)

When genuinely uncertain, answer "unsupported". Missing a false claim is worse
than flagging a true one.

Respond with exactly one word, then one sentence naming the specific claim.

- "supported"   — every factual claim traces to the sources
- "unsupported" — at least one claim does not

Label: """

JUDGE_RAILS = ["supported", "unsupported"]

# The judge runs on a different model from the agents, deliberately: a judge that
# shares the generator's blind spots is not an independent check.
#
# NOTE ON TEMPERATURE: gpt-5.6-terra accepts only its default temperature, so the
# judge cannot be pinned to 0.0 the way gpt-4o-mini was. It is not deterministic.
# Re-running the validation will move the numbers a little; a single TPR/TNR pair
# is a sample, not a fixed property of the prompt.
JUDGE_MODEL = "gpt-5.6-terra"


def format_sources(retrieved_chunks) -> str:
    """
    Render Agent 1's retrieved chunks as the judge's comparison set.
    Accepts ResearchResult.retrieved_chunks (list of dicts) or the already-joined
    CampaignStrategy.retrieved_facts string that Agent 2 propagates forward.
    """
    if isinstance(retrieved_chunks, str):
        return retrieved_chunks.strip() or "(no documents retrieved)"
    parts = []
    for chunk in retrieved_chunks or []:
        if isinstance(chunk, dict):
            name = chunk.get("doc_name", "unknown")
            parts.append(f"[Source: {name}]\n{chunk.get('text', '')}")
        else:
            parts.append(str(chunk))
    return "\n\n---\n\n".join(parts) if parts else "(no documents retrieved)"


def campaign_text_for_check(strategy) -> str:
    """
    The publishable copy: tagline, concept, key messages. Risk flags are
    deliberately excluded — they quote the claims Agent 2 *rejected*, and
    scanning them would halt campaigns that correctly caught their own problems.
    """
    parts = [
        strategy.tagline or "",
        strategy.campaign_concept or "",
        " ".join(strategy.key_messages or []),
    ]
    return "\n".join(p for p in parts if p.strip()).strip()


def _judge_rows_classic(rows, model=None):
    """phoenix[evals] <= 11: llm_classify + OpenAIModel."""
    import pandas as pd
    from phoenix.evals import OpenAIModel, llm_classify

    df = pd.DataFrame(rows)
    result = llm_classify(
        dataframe=df,
        template=GROUNDING_JUDGE_TEMPLATE,
        model=model or OpenAIModel(model=JUDGE_MODEL),
        rails=JUDGE_RAILS,
        provide_explanation=True,
    )
    explanations = (
        result["explanation"].tolist()
        if "explanation" in result.columns
        else [""] * len(result)
    )
    return list(zip(result["label"].tolist(), [e or "" for e in explanations]))


def _judge_rows_modern(rows, model=None):
    """
    phoenix[evals] >= 20: llm_classify and OpenAIModel were removed in favour of
    ClassificationEvaluator. Same contract — same prompt, same two rails, same
    model, explanation included — expressed in the current API. Temperature is
    left at the model default; gpt-5.6-terra does not accept another value.
    """
    from phoenix.evals import LLM, ClassificationEvaluator

    evaluator = ClassificationEvaluator(
        name="claim_grounding",
        llm=model or LLM(provider="openai", model=JUDGE_MODEL),
        prompt_template=GROUNDING_JUDGE_TEMPLATE,
        choices=JUDGE_RAILS,
        include_explanation=True,
    )

    judged = []
    for row in rows:
        scores = evaluator.evaluate(row)
        score = scores[0]
        judged.append((score.label, score.explanation or ""))
    return judged


def judge_rows(rows, model=None):
    """
    Classify a batch of {"sources", "output"} dicts.
    Returns [(label, explanation), ...] in input order.
    """
    try:
        from phoenix.evals import llm_classify  # noqa: F401
    except ImportError:
        return _judge_rows_modern(rows, model)
    return _judge_rows_classic(rows, model)


def run_judge(output_text: str, sources_text: str, model=None) -> tuple[str, str]:
    """
    Classify one campaign text against its sources.

    Returns (label, explanation). Raises if phoenix or the API is unavailable —
    callers decide how to fail, and run_grounding_check fails closed.
    """
    return judge_rows([{"sources": sources_text, "output": output_text}], model=model)[0]


# ---------------------------------------------------------------------------
# Locating the source line a claim contradicts
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "to", "we",
    "with", "you", "your", "this", "all", "not", "was", "were",
}

# What to look for in the sources when the flagged text itself is a poor query.
# A competitor's name never appears in Verdant's own documents, but the rule
# prohibiting the comparison does.
_SEARCH_HINTS = {
    COMPETITOR_NAMED: "compare directly to competitors by name",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9%]+", (text or "").lower()) if w not in _STOPWORDS}


def find_contradicting_source(claim: str, sources, failure_type: str = None) -> str:
    """
    Return the line from the retrieved sources that most directly speaks to the
    flagged claim — usually the one the claim contradicts or overstates.

    Deliberately lexical and deterministic: this runs after a halt, to show a
    reviewer what to read. It is not a second judge, and it does not spend a
    model call. Returns "" when nothing in the retrieved documents addresses the
    claim at all, which is itself worth showing — it means retrieval, not just
    generation, came up short.
    """
    query = f"{claim} {_SEARCH_HINTS.get(failure_type, '')}"
    claim_tokens = _tokens(query)
    if not claim_tokens:
        return ""

    best_line, best_score = "", 0.0
    for raw_line in format_sources(sources).splitlines():
        line = raw_line.strip().lstrip("-•").strip()
        if len(line) < 15 or line.startswith("[Source:"):
            continue
        overlap = claim_tokens & _tokens(line)
        if not overlap:
            continue
        # Weight by token length: "sustainable" discriminates, "any" does not.
        score = float(sum(len(t) for t in overlap))
        # Lines stating a limit are what a reviewer needs to see first, but this
        # is a tiebreaker — it must not beat a line that simply matches better.
        if re.search(r"\bnot\b|prohibit|cannot|in process|pending|yet", line, re.I):
            score *= 1.25
        # A flagged statistic is answered by the line carrying the real figure.
        if failure_type == UNAPPROVED_STAT and re.search(r"\d", line):
            score *= 1.5
        if score > best_score:
            best_line, best_score = line, score

    return best_line


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

@dataclass
class GroundingCheckResult:
    passed: bool
    layer: str                          # deterministic | judge | none
    failure_type: Optional[str] = None  # PROHIBITED_PHRASE | UNAPPROVED_STAT | ...
    claim_flagged: str = ""             # the offending text
    reason: str = ""                    # human-readable, becomes halt_reason
    judge_called: bool = False
    judge_label: Optional[str] = None
    judge_explanation: str = ""


def run_deterministic_checks(
    text: str,
    upstream_hallucination_flag: bool = False,
) -> Optional[GroundingCheckResult]:
    """Layer 1. Returns a failing result, or None if all checks pass."""
    # 1a — prohibited phrase (reuses Agent 2's policy tool unchanged)
    policy = check_brand_policy(text)
    if policy["status"] == "PROHIBITED":
        phrase = re.search(r"'([^']+)'", policy["reason"])
        flagged = phrase.group(1) if phrase else text[:120]
        return GroundingCheckResult(
            passed=False,
            layer=LAYER_DETERMINISTIC,
            failure_type=PROHIBITED_PHRASE,
            claim_flagged=flagged,
            reason=(
                f'Pipeline halted: campaign text contains the prohibited claim "{flagged}". '
                "This is not a verified Verdant brand claim. No creative assets were generated."
            ),
        )

    # 1b — unapproved statistic
    stat = check_unapproved_stat(text)
    if stat:
        number, context = stat
        return GroundingCheckResult(
            passed=False,
            layer=LAYER_DETERMINISTIC,
            failure_type=UNAPPROVED_STAT,
            claim_flagged=f"{number} — “…{context}…”",
            reason=(
                f'Pipeline halted: campaign text cites the statistic "{number}", which is not in '
                f'Verdant\'s approved statistics. Context: "…{context}…". '
                "Verify the figure against the brand guide or remove it."
            ),
        )

    # 1c — competitor named
    competitor = check_competitor_named(text)
    if competitor:
        return GroundingCheckResult(
            passed=False,
            layer=LAYER_DETERMINISTIC,
            failure_type=COMPETITOR_NAMED,
            claim_flagged=competitor,
            reason=(
                f'Pipeline halted: campaign text names the competitor "{competitor}". '
                "The brand guide prohibits direct comparison to competitors by name."
            ),
        )

    # Agent 2's own prohibited-phrase heuristic catches a few strings the policy
    # tool does not ("all manufacturing is fair trade", "switch to"). Free signal,
    # so it belongs in layer 1 rather than after a judge call.
    if upstream_hallucination_flag:
        return GroundingCheckResult(
            passed=False,
            layer=LAYER_DETERMINISTIC,
            failure_type=PROHIBITED_PHRASE,
            claim_flagged=text[:200],
            reason=(
                "Pipeline halted: Agent 2 flagged prohibited brand claims in this campaign "
                "strategy. No creative assets were generated from unverified content."
            ),
        )

    return None


class _NullSpans:
    """No-op span factory. Keeps the cascade traceable without requiring OTel."""

    @contextmanager
    def deterministic(self):
        yield None

    @contextmanager
    def judge(self):
        yield None


class TracedSpans:
    """
    Emits the child spans the pipeline's `verify` step needs, from inside the
    cascade — so the ordering and the fail-closed branch live in exactly one
    place. Pass an instance as `spans=` to run_grounding_check().
    """

    def __init__(self, tracer, span_kinds):
        self._tracer = tracer
        self._kinds = span_kinds  # {"tool": ..., "llm": ...} — values are strings

    @contextmanager
    def _span(self, name, kind):
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute("openinference.span.kind", kind)
            yield span

    @contextmanager
    def deterministic(self):
        with self._span("deterministic-checks", self._kinds["tool"]) as span:
            yield span

    @contextmanager
    def judge(self):
        with self._span("grounding-judge", self._kinds["llm"]) as span:
            yield span


def run_grounding_check(
    campaign_text: str,
    retrieved_chunks,
    model=None,
    upstream_hallucination_flag: bool = False,
    spans=None,
) -> GroundingCheckResult:
    """
    Full cascade: deterministic checks, then the judge on whatever survives.
    Deterministic failures never reach the judge — that's the cost saving,
    and `judge_called` on the span makes it measurable.

    `spans` optionally supplies context managers for the two layers, so a caller
    that traces gets child spans without a second copy of this ordering logic.
    Defaults to a no-op, so untraced callers behave exactly as before.
    """
    spans = spans or _NullSpans()

    with spans.deterministic() as det_span:
        deterministic = run_deterministic_checks(campaign_text, upstream_hallucination_flag)
        if det_span is not None:
            det_span.set_attribute("tool.name", "deterministic-checks")
            det_span.set_attribute(
                "output.value",
                deterministic.failure_type if deterministic else "PASSED",
            )
    if deterministic is not None:
        return deterministic

    sources_text = format_sources(retrieved_chunks)

    try:
        with spans.judge() as judge_span:
            label, explanation = run_judge(campaign_text, sources_text, model=model)
            if judge_span is not None:
                judge_span.set_attribute("llm.model_name", JUDGE_MODEL)
                judge_span.set_attribute("input.value", campaign_text)
                judge_span.set_attribute("output.value", f"{label}: {explanation}")
    except Exception as exc:
        # Fail closed: an unavailable judge is not evidence of grounding.
        return GroundingCheckResult(
            passed=False,
            layer=LAYER_JUDGE,
            failure_type=JUDGE_UNAVAILABLE,
            claim_flagged="",
            reason=(
                f"Pipeline halted: the grounding judge could not run ({type(exc).__name__}: {exc}). "
                "Campaign text was not verified against retrieved sources, so it is held for human review."
            ),
            judge_called=True,
        )

    if label == "unsupported":
        return GroundingCheckResult(
            passed=False,
            layer=LAYER_JUDGE,
            failure_type=UNSUPPORTED_CLAIM,
            claim_flagged=explanation.strip(),
            reason=(
                "Pipeline halted: a factual claim in the campaign is not supported by the brand "
                f"documents the pipeline retrieved. Judge: {explanation.strip()}"
            ),
            judge_called=True,
            judge_label=label,
            judge_explanation=explanation,
        )

    return GroundingCheckResult(
        passed=True,
        layer=LAYER_NONE,
        judge_called=True,
        judge_label=label,
        judge_explanation=explanation,
    )
