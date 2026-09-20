"""
Retrieval eval — did search find the right chunks?
==================================================

Deterministic. No judge, no generation. The only spend is the query embedding,
a fraction of a cent for the whole set.

This measures the failure mode behind most of the guardrail's flags. On
2026-09-19 the judge flagged four campaigns for claiming "87% from certified
sustainable sources" — and it was right, because the only retrieved line
mentioning 87% came from the sustainability report's "WHAT WE DO NOT CLAIM"
section ("100% sustainable materials (we're at 87%)"). The line that actually
says "87% of materials by weight are certified sustainable" was never retrieved.
Nothing in the eval suite could see that. This script can.

Ground truth is the golden dataset's `ground_truth_answer_contains` column: the
facts a correct answer to that brief must rest on. If they are not in the
retrieved chunks, generation is working blind no matter how good the model is.

    python3 -m evals.eval_retrieval
    python3 -m evals.eval_retrieval --samples 5

MATCHING RULE: an expected fact counts as retrieved when at least 60% of its
content tokens appear in the retrieved text. Substring matching is too strict
("45,000+ garments" never appears verbatim; the report says "45,312 garments
collected"), exact-set matching too loose. Every miss is printed in full so the
rule is auditable rather than trusted.
"""

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from pipeline.retrieval import (DEFAULT_N_RESULTS, derive_research_query,
                                expand_query, retrieve)

TOKEN_COVERAGE_THRESHOLD = 0.60

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "to", "we",
    "with", "not", "yet", "all", "any",
}


def _tokens(text: str) -> list:
    raw = re.findall(r"[a-z0-9%+\-]+", (text or "").lower())
    return [t for t in raw if t not in _STOPWORDS and len(t) > 1]


def _fact_found(fact: str, haystack_tokens: set) -> tuple[bool, list]:
    """Return (found, missing_tokens) for one expected fact."""
    toks = _tokens(fact)
    if not toks:
        return True, []
    missing = [t for t in toks if t not in haystack_tokens]
    coverage = 1.0 - (len(missing) / len(toks))
    return coverage >= TOKEN_COVERAGE_THRESHOLD, missing


def split_expected(raw: str) -> list:
    """
    Split on comma-space, not bare comma — "45,000+ garments, recycled, donated"
    must stay three facts, not four.
    """
    return [f.strip() for f in str(raw).split(", ") if f.strip()]


def run_retrieval_eval(sample_size: int = 20, n_results: int = DEFAULT_N_RESULTS,
                       expand: bool = False):
    print(f"\n{'='*70}")
    print("RETRIEVAL EVAL — did search find the chunks the brief needs?")
    print(f"{'='*70}\n")

    dataset = Path(__file__).parent.parent / "data" / "golden_dataset.csv"
    df = pd.read_csv(dataset).head(sample_size)
    print(f"✓ {len(df)} briefs · top-{n_results} retrieval · deterministic, no judge\n")

    rows, all_misses = [], []

    for _, row in df.iterrows():
        expansion_usd = 0.0
        if expand:
            query, exp_cost = expand_query(row["campaign_brief"])
            expansion_usd = exp_cost.usd or 0.0
            print(f"      expanded: {query[:100]}")
        else:
            query = derive_research_query(row["campaign_brief"])
        result = retrieve(query, n_results=n_results)

        haystack = " ".join(c["text"] for c in result.chunks)
        haystack_tokens = set(_tokens(haystack))

        expected = split_expected(row["ground_truth_answer_contains"])
        found, missed = [], []
        for fact in expected:
            ok, missing_tokens = _fact_found(fact, haystack_tokens)
            (found if ok else missed).append(fact)

        recall = len(found) / len(expected) if expected else 1.0
        rows.append({
            "id": row["id"],
            "brief": row["campaign_brief"],
            "derived_query": query,
            "n_chunks": len(result.chunks),
            "grounding_score": result.grounding_score,
            "sources": ", ".join(result.sources),
            "expected_facts": len(expected),
            "facts_found": len(found),
            "recall": round(recall, 3),
            "complete": len(missed) == 0,
            "missed_facts": " | ".join(missed),
            "expansion_usd": expansion_usd,
        })
        if missed:
            all_misses.append((row["id"], row["campaign_brief"], missed, result))

        flag = "✓" if not missed else "✗"
        print(f"  [{flag}] id {row['id']:>2}  recall {recall:>5.0%}  "
              f"({len(found)}/{len(expected)} facts)  grounding {result.grounding_score:.2f}")

    results = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "evals" / "retrieval_eval.csv"
    results.to_csv(out, index=False)

    mean_recall = results["recall"].mean()
    complete = int(results["complete"].sum())

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Mean fact recall @{n_results}:     {mean_recall:.1%}")
    print(f"  Briefs with every fact:    {complete}/{len(results)}")
    print(f"  Briefs missing something:  {len(results) - complete}/{len(results)}")
    print(f"  Total facts missed:        {int((results['expected_facts'] - results['facts_found']).sum())}")

    # Grounding score vs actual recall — the point being that the score does not
    # know what it failed to find.
    high = results[results["grounding_score"] >= 0.70]
    if len(high):
        print(f"\n  Of {len(high)} briefs the grounding score called 'high' (>=0.70), "
              f"{int((~high['complete']).sum())} were missing at least one required fact.")
        print("  The score measures whether retrieval found something relevant-looking,")
        print("  not whether it found what the brief actually needs.")

    if all_misses:
        print(f"\n{'-'*70}")
        print(f"MISSES ({len(all_misses)} briefs) — read these by hand")
        print(f"{'-'*70}")
        for rid, brief, missed, result in all_misses:
            print(f"\n  id {rid} · {brief}")
            print(f"  missing: {', '.join(missed)}")
            print(f"  retrieved {len(result.chunks)} chunks from {', '.join(result.sources)}:")
            for c in result.chunks:
                first = c["text"].strip().splitlines()[0][:88]
                print(f"      [{c['relevance_score']:.2f}] {first}")
            print(f"  {'-'*66}")

    if "expansion_usd" in results and results["expansion_usd"].sum():
        tot = results["expansion_usd"].sum()
        print(f"\n  Query expansion cost: ${tot:.5f} over {len(results)} briefs "
              f"= ${tot/len(results):.6f} per campaign")

    print(f"\n✓ Saved to evals/retrieval_eval.csv\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--expand", action="store_true",
                        help="EXPERIMENT B: expand the brief into search terms first")
    parser.add_argument("--n-results", type=int, default=DEFAULT_N_RESULTS,
                        help="chunks retrieved per query (tracks the pipeline default)")
    args = parser.parse_args()
    run_retrieval_eval(sample_size=args.samples, n_results=args.n_results,
                       expand=args.expand)
