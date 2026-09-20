"""
End-to-end eval — briefs in, campaigns out. Does it publish something bad?
==========================================================================

Runs the real pipeline. The number that matters is ESCAPES: campaigns that were
routed to publish while still containing a claim the brief's own row lists as
prohibited. An escape is a campaign a human would have had to catch.

    python3 -m evals.eval_end_to_end
    python3 -m evals.eval_end_to_end --samples 5 --assets

This is not the judge eval. Its inputs move every time the generator changes, so
its numbers are a snapshot of the current system, not a fixed measurement of the
judge. For that, see evals/eval_judge.py, which runs frozen text forever.

Sampling is disabled (sample_rate=0) so a run is reproducible; the audit sample
is a production behaviour, not an evaluation one. Assets are off by default —
video costs real money and tells us nothing about whether the copy was sound.
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from pipeline.run import SOURCE_EVAL, run_campaign_pipeline


def _split_prohibited(raw: str) -> list:
    return [p.strip().lower() for p in str(raw).split(", ") if p.strip()
            and not p.strip().startswith("(")]


def run_end_to_end(sample_size: int = 20, generate_assets: bool = False):
    print(f"\n{'='*70}")
    print("END-TO-END EVAL — does the pipeline publish anything it shouldn't?")
    print(f"{'='*70}\n")

    df = pd.read_csv(Path(__file__).parent.parent / "data" / "golden_dataset.csv").head(sample_size)
    print(f"✓ {len(df)} briefs · assets {'ON' if generate_assets else 'off'} · audit sampling off\n")

    rows, escapes = [], []

    for _, row in df.iterrows():
        brief = row["campaign_brief"]
        print(f"  [{row['id']:>2}] {brief[:52]}")
        t0 = time.time()
        try:
            run = run_campaign_pipeline(
                brief=brief,
                cycle=1,
                sample_rate=0.0,
                generate_assets=generate_assets,
                source=SOURCE_EVAL,
            )
        except Exception as e:
            print(f"       ❌ {type(e).__name__}: {e}")
            rows.append({"id": row["id"], "brief": brief, "error": str(e)})
            continue

        text = run.draft.verified_text().lower()
        prohibited = _split_prohibited(row["prohibited_claims"])
        hits = [p for p in prohibited if p in text]

        escaped = run.published and bool(hits)
        if escaped:
            escapes.append((row["id"], brief, hits, run))

        rows.append({
            "id": row["id"],
            "brief": brief,
            "status": run.final_status,
            "published": run.published,
            "caught_by": run.check_layer,
            "failure_type": run.failure_type or "",
            "flagged_claim": (run.flagged_claim or "")[:90],
            "repair_attempted": run.repair_attempted,
            "repair_fixed": run.repair_fixed,
            "n_claims": len(run.draft.claims),
            "receipts_without_source": sum(
                1 for r in run.routing.receipts if not r["source_found"]
            ),
            "grounding_score": run.retrieval.grounding_score,
            "prohibited_hits": "; ".join(hits),
            "escaped": escaped,
            "tokens": run.cost.total_tokens,
            "usd": run.cost.total_usd,
            "seconds": round(time.time() - t0, 1),
        })

        mark = "🚨 ESCAPE" if escaped else ("✅ published" if run.published else "🛑 flagged")
        detail = f" ({run.failure_type})" if run.failure_type else ""
        repair = " · repaired" if run.repair_attempted else ""
        print(f"       {mark}{detail}{repair} · {time.time()-t0:.0f}s")

    results = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "evals" / "end_to_end_results.csv"
    results.to_csv(out, index=False)

    clean = results[results.get("error").isna()] if "error" in results else results
    n = len(clean)
    published = int(clean["published"].sum())
    n_escapes = int(clean["escaped"].sum())
    repaired = int(clean["repair_attempted"].sum())
    fixed = int(clean["repair_fixed"].sum())
    priced = clean["usd"].dropna()

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Briefs run:            {n}")
    print(f"  Published:             {published}/{n}")
    print(f"  Flagged for review:    {n - published}/{n}")
    print(f"  Repairs attempted:     {repaired} · fixed {fixed}")
    print(f"  Claims with no source: {int(clean['receipts_without_source'].sum())}")
    print(f"\n  ESCAPES:               {n_escapes}/{n}", end="")
    print("  ← published with a prohibited claim" if n_escapes else "  ✅ nothing bad published")
    n_unknown = int(clean["usd"].isna().sum()) if "usd" in clean else 0
    if len(priced):
        print(f"\n  Cost: ${priced.sum():.4f} across {len(priced)} priced runs · "
              f"${priced.mean():.5f} mean")
        if n_unknown:
            print(f"        {n_unknown} run(s) UNKNOWN — usage not captured, excluded "
                  f"from the total rather than counted as zero")
    elif n_unknown:
        print(f"\n  Cost: UNKNOWN for all {n_unknown} run(s) — usage could not be captured")
    if n:
        print(f"  Time: {clean['seconds'].sum():.0f}s total · {clean['seconds'].mean():.0f}s mean")

    if escapes:
        print(f"\n{'-'*70}")
        print("ESCAPES IN FULL — every one of these is a miss")
        print(f"{'-'*70}")
        for rid, brief, hits, run in escapes:
            print(f"\n  id {rid} · {brief}")
            print(f"  prohibited phrase present: {', '.join(hits)}")
            print(f"  verify said: {run.final_status} (layer {run.check_layer})")
            print("  published text:")
            for line in run.draft.verified_text().splitlines():
                print(f"      {line}")
            print(f"  {'-'*66}")

    by_type = clean[clean["failure_type"] != ""]["failure_type"].value_counts()
    if len(by_type):
        print(f"\n  What caught the flags:")
        for ftype, count in by_type.items():
            print(f"    {ftype:22} {count}")

    print(f"\n✓ Saved to evals/end_to_end_results.csv\n")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--assets", action="store_true", help="also generate video (costs money)")
    args = parser.parse_args()
    run_end_to_end(sample_size=args.samples, generate_assets=args.assets)
