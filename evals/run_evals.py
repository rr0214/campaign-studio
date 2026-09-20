"""
DEPRECATED — this script did two different jobs in one pass, and that was the
problem. It has been split:

    python3 -m evals.eval_retrieval    Did search find the right chunks?
                                       Deterministic, free, no judge.

    python3 -m evals.eval_end_to_end   Briefs in, campaigns out — does the
                                       pipeline publish anything bad? Runs the
                                       real pipeline; inputs move as it changes.

    python3 -m evals.eval_judge        Frozen text in, verdict out — does the
                                       judge agree with a fixed label? No
                                       pipeline. Same cases forever.

Mixing the last two is how the 2026-09-19 run reported TPR 0.00 for a judge that
turned out to be right on every disputed row: it scored generated output against
`expected_hallucination`, a column describing how risky a *query* looked rather
than whether the *text* contained a false claim.
"""

import sys

BANNER = __doc__


def main():
    print(BANNER)
    print("Nothing was run. Pick one of the three above.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
