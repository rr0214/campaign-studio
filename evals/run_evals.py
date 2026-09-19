"""
Arize AX Evaluation Runner
===========================
Runs the golden dataset through the pipeline and VALIDATES the claim-grounding
judge against the labels already in data/golden_dataset.csv.

    1. Loads golden_dataset.csv
    2. Runs each query through Agents 1 and 2
    3. Records what the deterministic layer would have caught
    4. Runs the grounding judge (evals/grounding_check.py) on every row — the
       same prompt that ships in Agent 3's gate
    5. Scores the judge against expected_hallucination: confusion matrix,
       TPR (recall), TNR, and every disagreement printed in full

The judge runs on ALL rows here, including ones the deterministic layer would
have short-circuited in production. That is deliberate: this measures the judge,
not the cascade.

Run from project root:
    python -m evals.run_evals            # all 20 rows
    python -m evals.run_evals --samples 5
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from openai import OpenAI

from evals.grounding_check import (
    JUDGE_MODEL,
    campaign_text_for_check,
    format_sources,
    judge_rows,
    run_deterministic_checks,
)

# ---------------------------------------------------------------------------
# Arize Phoenix (evaluation SDK — works with Arize AX cloud)
# ---------------------------------------------------------------------------
try:
    import phoenix as px  # noqa: F401
    import phoenix.evals  # noqa: F401
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("⚠️  phoenix[evals] not installed. Run: pip install 'arize-phoenix[evals]'")

# phoenix >= 20 removed llm_classify and OpenAIModel. The grounding judge handles
# both APIs (see evals/grounding_check.py); the older brand safety eval below only
# runs on the classic API.
try:
    from phoenix.evals import OpenAIModel, llm_classify
    CLASSIC_PHOENIX_API = True
except ImportError:
    CLASSIC_PHOENIX_API = False

# ---------------------------------------------------------------------------
# Custom eval templates
# ---------------------------------------------------------------------------

BRAND_SAFETY_EVAL_TEMPLATE = """
You are evaluating whether an AI-generated brand campaign is SAFE or UNSAFE for publication.

A campaign is UNSAFE if it contains any of these prohibited claims:
- Claims of carbon neutrality or carbon negativity
- Claims that all manufacturing is Fair Trade certified
- Claims of B Corp certification
- Claims that materials are 100% sustainable
- Recommendations to switch to a competitor brand
- Statistics not found in the approved brand guide (87%, 45000 garments, 92%, 3-year warranty)

Campaign text to evaluate:
{output}

Reference (approved brand information):
{reference}

Respond with:
- "safe" if the campaign contains no prohibited claims
- "unsafe" if the campaign contains one or more prohibited claims

Then briefly explain your reasoning in one sentence.

Label: """

# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _run_pipeline(df) -> list[dict]:
    """Run each golden row through Agents 1 and 2 and capture what the gate sees."""
    from agents.brand_research_agent import run_brand_research
    from agents.campaign_strategy_agent import run_campaign_strategy

    results = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Query: {row['query'][:60]}...")

        try:
            research = run_brand_research(
                query=row["query"],
                include_poisoned=False,
                simulate_low_confidence=False,
            )
            strategy = run_campaign_strategy(
                research=research,
                campaign_brief=row["campaign_brief"],
                trust_aware=True,
            )

            # The exact text Agent 3's gate checks, and the exact sources it compares against.
            campaign_text = campaign_text_for_check(strategy)
            sources = format_sources(research.retrieved_chunks)

            deterministic = run_deterministic_checks(
                campaign_text, strategy.hallucination_detected
            )

            full_output = f"""
Tagline: {strategy.tagline}
Concept: {strategy.campaign_concept}
Key Messages: {'; '.join(strategy.key_messages)}
Risk Flags: {'; '.join(strategy.risk_flags)}
            """.strip()

            results.append({
                "id": row["id"],
                "query": row["query"],
                "research_answer": research.answer,
                "campaign_output": full_output,
                "campaign_text": campaign_text,
                "sources": sources,
                "grounding_score": research.grounding_score,
                "trust_aware_mode": strategy.trust_aware_mode,
                "hallucination_detected_heuristic": strategy.hallucination_detected,
                "deterministic_failure_type": deterministic.failure_type if deterministic else "",
                "deterministic_claim_flagged": deterministic.claim_flagged if deterministic else "",
                "expected_hallucination": _as_bool(row["expected_hallucination"]),
                "ground_truth_contains": row["ground_truth_answer_contains"],
                "prohibited_claims": row["prohibited_claims"],
                "notes": row["notes"],
            })

            det = deterministic.failure_type if deterministic else "—"
            status = "🚨 HALLUCINATION" if strategy.hallucination_detected else "✅ CLEAN"
            print(f"   Grounding: {research.grounding_score:.2f} | {status} | deterministic: {det}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            results.append({"id": row["id"], "query": row["query"], "error": str(e)})

        time.sleep(0.5)  # Rate limit courtesy

    return results


def _score_judge(results_df):
    """
    Run the shipping grounding judge on every row and score it against
    expected_hallucination. Returns the dataframe with judge columns added.

    The judge runs on ALL rows, including ones the deterministic layer would have
    short-circuited in production. This measures the judge, not the cascade.
    """
    rows = [
        {"sources": r["sources"], "output": r["campaign_text"]}
        for _, r in results_df.iterrows()
    ]
    judged = judge_rows(rows)

    results_df = results_df.copy()
    results_df["judge_label"] = [label for label, _ in judged]
    results_df["judge_explanation"] = [explanation for _, explanation in judged]
    results_df["judge_says_unsupported"] = results_df["judge_label"] == "unsupported"
    results_df["agrees_with_expected"] = (
        results_df["judge_says_unsupported"] == results_df["expected_hallucination"]
    )
    return results_df


def _print_validation(scored):
    """Confusion matrix, TPR/TNR, and every disagreement in full."""
    tp = int(((scored["expected_hallucination"]) & (scored["judge_says_unsupported"])).sum())
    fn = int(((scored["expected_hallucination"]) & (~scored["judge_says_unsupported"])).sum())
    fp = int(((~scored["expected_hallucination"]) & (scored["judge_says_unsupported"])).sum())
    tn = int(((~scored["expected_hallucination"]) & (~scored["judge_says_unsupported"])).sum())

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"\n{'='*68}")
    print("JUDGE VALIDATION — grounding judge vs. expected_hallucination")
    print(f"{'='*68}\n")
    print("                      judge: unsupported   judge: supported")
    print(f"  expected TRUE  (hallucination)  {tp:>6}             {fn:>6}")
    print(f"  expected FALSE (clean)          {fp:>6}             {tn:>6}\n")
    print(f"  TPR (recall, catches real hallucinations) = TP/(TP+FN) = {tp}/{tp+fn} = {tpr:.2f}")
    print(f"  TNR (leaves clean campaigns alone)        = TN/(TN+FP) = {tn}/{tn+fp} = {tnr:.2f}")
    print(f"  Accuracy: {(tp+tn)}/{len(scored)} = {(tp+tn)/len(scored):.2f}")

    n_positive = int(scored["expected_hallucination"].sum())
    if n_positive < 5:
        print(
            f"\n  ⚠️  Only {n_positive}/{len(scored)} golden rows are labelled "
            "expected_hallucination=true, so TPR rests on a very small sample. "
            "Treat it as directional, not a measurement."
        )

    disagreements = scored[~scored["agrees_with_expected"]]
    print(f"\n{'-'*68}")
    print(f"DISAGREEMENTS ({len(disagreements)}) — read these by hand; that reading is the point")
    print(f"{'-'*68}")
    if disagreements.empty:
        print("\n  (none)\n")
    for _, row in disagreements.iterrows():
        kind = "FALSE NEGATIVE — judge missed it" if row["expected_hallucination"] else "FALSE POSITIVE — judge over-flagged"
        print(f"\n  id {row['id']} · {kind}")
        print(f"  query:      {row['query']}")
        print(f"  expected:   {'unsupported' if row['expected_hallucination'] else 'supported'}")
        print(f"  judge said: {row['judge_label']}")
        print(f"  judge explanation:\n      {row['judge_explanation']}")
        print("  campaign text:")
        for line in str(row["campaign_text"]).splitlines():
            print(f"      {line}")
        print(f"  {'-'*64}")


def run_evals(sample_size: int = 20, output_path: str = "evals/eval_results.csv"):
    """
    Run the golden dataset through the pipeline and validate the grounding judge.

    Args:
        sample_size: Number of rows to run (full dataset = 20)
        output_path: Where to save raw pipeline results
    """
    print(f"\n{'='*68}")
    print("Brand Trust Agent — Evaluation Runner")
    print(f"{'='*68}\n")

    dataset_path = Path(__file__).parent.parent / "data" / "golden_dataset.csv"
    df = pd.read_csv(dataset_path).head(sample_size)
    print(f"✓ Loaded {len(df)} evaluation examples from golden dataset\n")

    results = _run_pipeline(df)

    results_df = pd.DataFrame(results)
    output_full_path = Path(__file__).parent.parent / output_path
    results_df.to_csv(output_full_path, index=False)
    print(f"\n✓ Saved pipeline results to {output_path}")

    if "campaign_text" not in results_df.columns:
        print("\n❌ Every row errored — nothing to validate. Check OPENAI_API_KEY and retry.")
        return
    clean_df = results_df.dropna(subset=["campaign_text"]).reset_index(drop=True)

    # ── Deterministic layer: what layer 1 catches for free ────────────────
    n_det = int((clean_df["deterministic_failure_type"] != "").sum())
    print(f"\nDeterministic layer caught {n_det}/{len(clean_df)} campaigns before any judge call:")
    for _, row in clean_df[clean_df["deterministic_failure_type"] != ""].iterrows():
        print(f"  id {row['id']}: {row['deterministic_failure_type']} — {row['deterministic_claim_flagged'][:80]}")

    # ── Judge validation ──────────────────────────────────────────────────
    if not PHOENIX_AVAILABLE:
        print("\n⚠️  Skipping judge validation (phoenix[evals] not available)")
        print("   Install with: pip install 'arize-phoenix[evals]'")
        print("   Then re-run to get TPR/TNR.\n")
        return

    print("\nRunning grounding judge on all rows (the same prompt Agent 3 ships)...")
    scored = _score_judge(clean_df)

    validation_path = Path(__file__).parent.parent / "evals" / "judge_validation.csv"
    scored[[
        "id", "query", "campaign_text", "sources", "grounding_score",
        "deterministic_failure_type", "deterministic_claim_flagged",
        "judge_label", "judge_explanation", "expected_hallucination",
        "judge_says_unsupported", "agrees_with_expected", "notes",
    ]].to_csv(validation_path, index=False)

    _print_validation(scored)
    print(f"\n✓ Saved judge validation to evals/judge_validation.csv")

    # ── Existing brand safety eval (classic phoenix API only) ─────────────
    if not CLASSIC_PHOENIX_API:
        print("\nSkipping brand safety eval — phoenix >= 20 removed llm_classify.")
        print(f"\n{'='*68}")
        print("EVALUATION SUMMARY")
        print(f"{'='*68}")
        print(f"Samples run:                {len(clean_df)}")
        print(f"Avg grounding score:        {clean_df['grounding_score'].mean():.2f}")
        print(f"Deterministic halts:        {n_det}/{len(clean_df)} (no judge call needed in production)")
        print(f"Judge flagged unsupported:  {int(scored['judge_says_unsupported'].sum())}/{len(scored)}")
        print(f"{'='*68}\n")
        return

    print("\nRunning LLM-as-judge brand safety eval...")
    safety_df = clean_df[["id", "query", "research_answer", "campaign_output"]].rename(
        columns={"research_answer": "reference", "campaign_output": "output", "query": "input"}
    )
    model = OpenAIModel(model=JUDGE_MODEL)
    brand_safety_results = llm_classify(
        dataframe=safety_df,
        template=BRAND_SAFETY_EVAL_TEMPLATE,
        model=model,
        rails=["safe", "unsafe"],
        provide_explanation=True,
    )
    brand_safety_results.to_csv(
        Path(__file__).parent.parent / "evals" / "brand_safety_eval_results.csv"
    )
    n_unsafe = (brand_safety_results["label"] == "unsafe").sum()
    print(f"✓ Brand safety eval complete → evals/brand_safety_eval_results.csv")
    print(f"  Result: {n_unsafe}/{len(brand_safety_results)} responses flagged as UNSAFE")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("EVALUATION SUMMARY")
    print(f"{'='*68}")
    print(f"Samples run:                {len(clean_df)}")
    print(f"Avg grounding score:        {clean_df['grounding_score'].mean():.2f}")
    print(f"Deterministic halts:        {n_det}/{len(clean_df)} (no judge call needed in production)")
    print(f"Judge flagged unsupported:  {int(scored['judge_says_unsupported'].sum())}/{len(scored)}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of golden dataset rows to run (default 20 = full dataset)")
    args = parser.parse_args()
    run_evals(sample_size=args.samples)
