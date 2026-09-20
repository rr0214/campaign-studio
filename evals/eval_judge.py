"""
Judge eval — frozen text in, verdict out.
=========================================

Does the grounding judge agree with a hand-assigned label, on text that never
changes? No pipeline, no retrieval, no generation. The same sixteen cases
forever, so a judge or prompt change is measured against a fixed target.

    python3 -m evals.eval_judge

WHY THIS IS SEPARATE FROM THE END-TO-END EVAL
The two ask different questions and must not share a run. End-to-end asks "does
the system publish something bad", and its inputs move every time the generator
does. This asks "does the judge call this text correctly", and its inputs must
never move. Mixing them — one script, one pass, one set of rows — is how the
2026-09-19 run ended up reporting TPR 0.00 for a judge that was right every time:
it was scored against `expected_hallucination`, a column describing how risky a
*query* looked, not whether the *generated text* contained a false claim.

THE LABELS IN data/judge_eval_set.csv ARE HAND-ASSIGNED AND NEED REVIEW.
They were written by Claude against the brand documents, not by a brand expert.
Case 13 in particular is a deliberate judgement call and is flagged in its own
rationale field. Read them before trusting any number this prints.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from evals.grounding_check import JUDGE_MODEL, judge_rows

FIXTURE = Path(__file__).parent.parent / "data" / "judge_eval_set.csv"


def run_judge_eval(only_judge_reachable: bool = False):
    print(f"\n{'='*70}")
    print("JUDGE EVAL — frozen text, fixed labels")
    print(f"{'='*70}\n")

    df = pd.read_csv(FIXTURE)
    if only_judge_reachable:
        df = df[df["deterministic_catch"] == "no"].reset_index(drop=True)
        print("Restricted to cases the deterministic layer does NOT short-circuit —\n"
              "i.e. the cases the judge actually sees in production.\n")

    print(f"✓ {len(df)} frozen cases · judge = {JUDGE_MODEL}")
    print(f"  expected: {int((df['expected_label']=='unsupported').sum())} unsupported, "
          f"{int((df['expected_label']=='supported').sum())} supported\n")

    rows = [{"sources": r["sources"], "output": r["campaign_text"]} for _, r in df.iterrows()]
    judged = judge_rows(rows)

    df = df.copy()
    df["judge_label"] = [label for label, _ in judged]
    df["judge_explanation"] = [expl for _, expl in judged]
    df["correct"] = df["judge_label"] == df["expected_label"]

    tp = int(((df["expected_label"] == "unsupported") & (df["judge_label"] == "unsupported")).sum())
    fn = int(((df["expected_label"] == "unsupported") & (df["judge_label"] == "supported")).sum())
    fp = int(((df["expected_label"] == "supported") & (df["judge_label"] == "unsupported")).sum())
    tn = int(((df["expected_label"] == "supported") & (df["judge_label"] == "supported")).sum())

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    print("                          judge: unsupported   judge: supported")
    print(f"  label unsupported             {tp:>6}             {fn:>6}")
    print(f"  label supported               {fp:>6}             {tn:>6}\n")
    print(f"  TPR (catches bad text)  = {tp}/{tp+fn} = {tpr:.2f}")
    print(f"  TNR (leaves good alone) = {tn}/{tn+fp} = {tnr:.2f}")
    print(f"  Accuracy                = {tp+tn}/{len(df)} = {(tp+tn)/len(df):.2f}")

    wrong = df[~df["correct"]]
    print(f"\n{'-'*70}")
    print(f"DISAGREEMENTS ({len(wrong)}) — read by hand; that reading is the point")
    print(f"{'-'*70}")
    if wrong.empty:
        print("\n  (none)\n")
    for _, r in wrong.iterrows():
        kind = ("FALSE NEGATIVE — judge missed it" if r["expected_label"] == "unsupported"
                else "FALSE POSITIVE — judge over-flagged")
        print(f"\n  id {r['id']} · {r['case']} · {kind}")
        print(f"  expected:   {r['expected_label']}")
        print(f"  judge said: {r['judge_label']}")
        print(f"  judge explanation:\n      {r['judge_explanation']}")
        print("  campaign text:")
        for line in str(r["campaign_text"]).splitlines():
            print(f"      {line}")
        print("  sources shown to the judge:")
        for line in str(r["sources"]).splitlines():
            print(f"      {line}")
        print(f"  why it was labelled that way:\n      {r['rationale']}")
        print(f"  {'-'*66}")

    out = Path(__file__).parent.parent / "evals" / "judge_eval_results.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved to evals/judge_eval_results.csv")
    print("\n  Reminder: the judge runs at its model's default temperature "
          "(gpt-5.6-terra\n  accepts no other value), so these numbers move a little "
          "between runs.\n  Treat one run as a sample, not a fixed property of the prompt.\n")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-reachable-only", action="store_true",
                        help="Skip cases the deterministic layer catches first")
    args = parser.parse_args()
    run_judge_eval(only_judge_reachable=args.judge_reachable_only)
