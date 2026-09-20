"""
Video judge eval — does the visual-claim judge agree with your labels?
======================================================================

Frozen shot descriptions, fixed labels, no pipeline. The same six cases forever,
so a change to VIDEO_JUDGE_TEMPLATE is measured against a fixed target.

    python3 -m evals.eval_video_judge

THE LABELS IN data/video_judge_eval_set.csv ARE YOURS.
They were deliberately left empty. This script refuses to run until they are
filled, and it will not guess, infer or default them. The judge is being measured
against a human's reading of what a viewer would conclude — a label this file
wrote itself would make the measurement circular.

The question the video judge asks is not the text judge's question. Copy is
judged on what it says; a shot is judged on what an ordinary viewer would walk
away believing. A shot can assert something no sentence in the campaign claimed.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
from collections import Counter

import pandas as pd

from evals.grounding_check import JUDGE_MODEL, VIDEO_JUDGE_TEMPLATE, judge_rows
from pipeline.retrieval import derive_research_query, retrieve

FIXTURE = Path(__file__).parent.parent / "data" / "video_judge_eval_set.csv"
VALID = {"supported", "unsupported"}

# The shots are about materials, the factory and the Take Back Program, so the
# judge is given what retrieval returns for that ground rather than a curated
# set — the same comparison basis the runtime judge gets.
SOURCE_BRIEF = "Our materials, our factories and what happens to garments sent back"


def _load():
    df = pd.read_csv(FIXTURE)
    df["label"] = df["label"].astype("string").str.strip().str.lower()
    unlabelled = df[df["label"].isna() | (df["label"] == "")]
    if len(unlabelled):
        print(f"\n{'='*70}")
        print("NOT RUNNING — labels are not filled in")
        print(f"{'='*70}\n")
        print(f"{len(unlabelled)} of {len(df)} rows in data/video_judge_eval_set.csv have no")
        print("label. Fill the `label` column with 'supported' or 'unsupported'.\n")
        for _, r in unlabelled.iterrows():
            print(f"  id {r['id']}")
            print(f"    shot : {r['shot'][:96]}…")
            print(f"    notes: {r['notes'][:96]}…\n")
        print("These are your calls to make. This script will not assign them.\n")
        return None

    bad = df[~df["label"].isin(VALID)]
    if len(bad):
        print(f"\nInvalid labels (must be one of {sorted(VALID)}):")
        for _, r in bad.iterrows():
            print(f"  id {r['id']}: {r['label']!r}")
        return None
    return df


def run_stability(repeat: int):
    """
    Measure how consistently the judge returns the same verdict for the same
    shot. Deliberately label-free: it never reads the label column, never
    computes agreement, and never prints one.

    Stability and accuracy are different properties and tuning a prompt while
    watching accuracy is how you end up fitting the target. This measures only
    whether the judge says the same thing twice.
    """
    df = pd.read_csv(FIXTURE)
    print(f"\n{'='*70}")
    print(f"VIDEO JUDGE STABILITY — {repeat} runs per shot, labels not consulted")
    print(f"{'='*70}\n")
    print(f"  prompt sha256: {hashlib.sha256(VIDEO_JUDGE_TEMPLATE.encode()).hexdigest()[:16]}…")
    print(f"  judge: {JUDGE_MODEL} · {len(df)} shots · {len(df) * repeat} calls\n")

    retrieval = retrieve(derive_research_query(SOURCE_BRIEF))
    rows = [{"sources": retrieval.sources_text, "shot": r["shot"]} for _, r in df.iterrows()]

    runs = []
    for i in range(repeat):
        verdicts = [label for label, _ in judge_rows(rows, template=VIDEO_JUDGE_TEMPLATE)]
        runs.append(verdicts)
        print(f"  run {i+1}: {' '.join(v[:3] for v in verdicts)}")

    print(f"\n{'-'*70}")
    print("PER-CASE CONSISTENCY")
    print(f"{'-'*70}")
    stable = 0
    per_case = []
    for idx, row in df.iterrows():
        votes = [r[idx] for r in runs]
        top = Counter(votes).most_common(1)[0]
        consistency = top[1] / len(votes)
        per_case.append(consistency)
        if consistency == 1.0:
            stable += 1
        flag = "" if consistency == 1.0 else "   <- flips"
        spread = dict(Counter(votes))
        print(f"  id {row['id']}  {top[0]:<12} {consistency:>5.0%}  {spread}{flag}")

    overall = sum(per_case) / len(per_case)
    print(f"\n  Fully stable cases : {stable}/{len(df)}")
    print(f"  Mean consistency   : {overall:.0%}")
    print(f"\n  (No labels were read. Accuracy is measured separately, once, "
          f"after the prompt is frozen.)\n")
    return {"stable": stable, "total": len(df), "mean": overall}


def run_video_judge_eval():
    print(f"\n{'='*70}")
    print("VIDEO JUDGE EVAL — would a viewer conclude something unsupported?")
    print(f"{'='*70}\n")

    df = _load()
    if df is None:
        return None

    print(f"✓ {len(df)} frozen shots · judge = {JUDGE_MODEL}")
    print(f"  your labels: {int((df['label']=='unsupported').sum())} unsupported, "
          f"{int((df['label']=='supported').sum())} supported\n")

    retrieval = retrieve(derive_research_query(SOURCE_BRIEF))
    print(f"  comparison basis: {len(retrieval.chunks)} retrieved chunks from "
          f"{', '.join(retrieval.sources)}\n")

    rows = [{"sources": retrieval.sources_text, "shot": r["shot"]} for _, r in df.iterrows()]
    judged = judge_rows(rows, template=VIDEO_JUDGE_TEMPLATE)

    df = df.copy()
    df["judge_label"] = [l for l, _ in judged]
    df["judge_explanation"] = [e for _, e in judged]
    df["agrees"] = df["judge_label"] == df["label"]

    tp = int(((df["label"] == "unsupported") & (df["judge_label"] == "unsupported")).sum())
    fn = int(((df["label"] == "unsupported") & (df["judge_label"] == "supported")).sum())
    fp = int(((df["label"] == "supported") & (df["judge_label"] == "unsupported")).sum())
    tn = int(((df["label"] == "supported") & (df["judge_label"] == "supported")).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    print("                          judge: unsupported   judge: supported")
    print(f"  you: unsupported             {tp:>6}             {fn:>6}")
    print(f"  you: supported               {fp:>6}             {tn:>6}\n")
    print(f"  Agreement = {int(df['agrees'].sum())}/{len(df)} = {df['agrees'].mean():.2f}")
    print(f"  TPR (catches overclaiming shots) = {tp}/{tp+fn} = {tpr:.2f}")
    print(f"  TNR (leaves clean shots alone)   = {tn}/{tn+fp} = {tnr:.2f}")
    print(f"\n  Six cases is a smoke test, not a measurement. Read the disagreements.")

    wrong = df[~df["agrees"]]
    print(f"\n{'-'*70}")
    print(f"DISAGREEMENTS ({len(wrong)}) — in full")
    print(f"{'-'*70}")
    if wrong.empty:
        print("\n  (none)\n")
    for _, r in wrong.iterrows():
        kind = ("JUDGE MISSED IT — you called it unsupported"
                if r["label"] == "unsupported" else
                "JUDGE OVER-FLAGGED — you called it supported")
        print(f"\n  id {r['id']} · {kind}")
        print(f"  your label : {r['label']}")
        print(f"  judge said : {r['judge_label']}")
        print("  shot:")
        for line in str(r["shot"]).split(". "):
            if line.strip():
                print(f"      {line.strip()}.")
        print(f"  judge explanation:\n      {r['judge_explanation']}")
        print(f"  your note when the fixture was written:\n      {r['notes']}")
        print(f"  {'-'*66}")

    out = Path(__file__).parent.parent / "evals" / "video_judge_eval_results.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved to evals/video_judge_eval_results.csv")
    print("\n  The judge runs at its model's default temperature, so these numbers")
    print("  move between runs. Treat one run as a sample.\n")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=0,
                        help="Measure verdict stability over N runs. Label-free — "
                             "reads no labels and reports no agreement.")
    args = parser.parse_args()
    if args.repeat:
        run_stability(args.repeat)
        sys.exit(0)
    sys.exit(0 if run_video_judge_eval() is not None else 1)
