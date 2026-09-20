"""
Verdant brand policy — the approved and prohibited claim tables.
================================================================

Single home for the claim tables and the policy check. Lives at the repo root
because both the guardrail (evals/grounding_check.py) and the pipeline import it;
putting it under either would make one depend on the other.

THE TABLE IS SELF-CHECKING. Every approved claim carries the verbatim line from
data/brand_docs/ that it comes from, and that line is verified against the
documents at first use. An entry whose source cannot be located is not treated as
approved — check_brand_policy() will not stamp it verified, and
`python3 -m brand_policy` exits non-zero.

This exists because the table previously carried three claims nothing supported:
two about recycled packaging (no brand document mentions packaging at all) and
"45,000 garments diverted annually" (the source says 45,312 cumulative since
2021, roughly a fifth of the implied rate). The grounding judge caught campaign
copy repeating the packaging claim verbatim on rows 1, 10 and 18 of the
2026-09-19 validation run. A claim stamped "(verified)" that no document
supports is worse than an unchecked model, because the policy layer lends it a
credential the evidence does not.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

BRAND_DOCS_DIR = Path(__file__).parent / "data" / "brand_docs"


@dataclass(frozen=True)
class ApprovedClaim:
    """An approved claim and the document line it rests on."""
    text: str           # how the claim may be phrased in campaign copy
    source_quote: str   # verbatim substring of a brand document
    source_doc: str     # which document it came from

    def __str__(self) -> str:
        return self.text


APPROVED_CLAIMS = {
    # CORRECTED 2026-09-19: previously "45,000 garments diverted from landfill
    # annually". The source figure is cumulative since launch, not a yearly rate.
    "45000": ApprovedClaim(
        text="45,000+ garments collected through the Take Back Program since 2021, cumulative (verified)",
        source_quote="Since launch (2021): 45,312 garments collected",
        source_doc="verdant_sustainability.txt",
    ),
    "45,000": ApprovedClaim(
        text="45,000+ garments collected through the Take Back Program since 2021, cumulative (verified)",
        source_quote="Take Back Program: 45,000+ garments recycled",
        source_doc="verdant_brand_guide.txt",
    ),
    # CORRECTED 2026-09-19: added "by weight". Dropping the basis broadens the
    # claim, and the judge flagged copy that dropped it four times.
    "87%": ApprovedClaim(
        text="87% of materials by weight are certified sustainable (verified)",
        source_quote="87% of materials by weight are certified sustainable",
        source_doc="verdant_sustainability.txt",
    ),
    # CORRECTED 2026-09-19: previously "Uses recycled materials in packaging and
    # core product lines". No brand document mentions packaging. Narrowed to the
    # materials the documents actually name.
    "recycled": ApprovedClaim(
        text="Uses certified recycled polyester and recycled elastane in garments (verified)",
        source_quote="Certified Recycled Polyester: Sourced from certified post-consumer plastic bottles",
        source_doc="verdant_sustainability.txt",
    ),
    "performance": ApprovedClaim(
        text="Performance-grade sustainable activewear (verified)",
        source_quote="Verdant apparel meets or exceeds comparable non-sustainable athletic wear",
        source_doc="verdant_brand_guide.txt",
    ),
    # REMOVED 2026-09-19: "climate" → "Climate-conscious packaging initiative".
    # Wholly unsupported: no packaging initiative and no climate programme appears
    # in any brand document. Claims containing "climate" now fall through to
    # UNVERIFIED, which is the correct answer.
}

PROHIBITED_CLAIMS = [
    "carbon neutral", "carbon-neutral", "carbon negative", "net zero",
    "b corp", "b-corp", "b corp certified",
    "100% sustainable", "100% fair trade",
    "fair trade certified", "sri lanka fair trade",
    "ecoelite", "switch to ecoelite",
    "carbon offset",
]


# ---------------------------------------------------------------------------
# Self-check: can every approved claim be traced to a document?
# ---------------------------------------------------------------------------

_traceable_cache = None


def _load_docs() -> dict:
    docs = {}
    if BRAND_DOCS_DIR.exists():
        for path in sorted(BRAND_DOCS_DIR.glob("*.txt")):
            # The poisoned document is an attack fixture, not evidence. A claim
            # that can only be traced to it is not supported.
            if "poisoned" in path.name:
                continue
            docs[path.name] = path.read_text()
    return docs


def audit_approved_claims() -> list:
    """
    Check every approved claim against the brand documents.
    Returns a list of problem dicts — empty means the table is fully traceable.
    """
    docs = _load_docs()
    problems = []
    for key, claim in APPROVED_CLAIMS.items():
        if not isinstance(claim, ApprovedClaim):
            problems.append({"key": key, "issue": "not an ApprovedClaim — no source recorded"})
            continue
        if not claim.source_quote.strip():
            problems.append({"key": key, "issue": "empty source_quote"})
            continue
        named = docs.get(claim.source_doc)
        if named is None:
            problems.append({"key": key, "issue": f"source_doc '{claim.source_doc}' not found"})
            continue
        if claim.source_quote not in named:
            found_in = [name for name, text in docs.items() if claim.source_quote in text]
            problems.append({
                "key": key,
                "issue": (f"source_quote not found verbatim in {claim.source_doc}"
                          + (f" (it is in {', '.join(found_in)})" if found_in
                             else " — not in any brand document")),
                "quote": claim.source_quote,
            })
    return problems


def _traceable_keys() -> set:
    """Keys whose source line was located. Computed once, then cached."""
    global _traceable_cache
    if _traceable_cache is None:
        bad = {p["key"] for p in audit_approved_claims()}
        _traceable_cache = set(APPROVED_CLAIMS) - bad
    return _traceable_cache


def check_brand_policy(claim: str) -> dict:
    claim_lower = claim.lower()

    for prohibited in PROHIBITED_CLAIMS:
        if prohibited in claim_lower:
            return {
                "status": "PROHIBITED",
                "claim_checked": claim,
                "reason": f"'{prohibited}' is not a verified Verdant brand claim.",
                # This string is returned to the model as tool output, so anything
                # fabricated here is actively suggested back to it. It previously
                # recommended "climate-conscious packaging" — the exact unsupported
                # claim removed from the table above.
                "approved_alternative": (
                    "Use only verified claims: 87% of materials by weight certified "
                    "sustainable, 45,000+ garments collected since 2021, certified "
                    "recycled polyester."
                ),
            }

    traceable = _traceable_keys()
    for key, approved in APPROVED_CLAIMS.items():
        if key.lower() in claim_lower:
            if key not in traceable:
                # The entry exists but its source could not be located. Not
                # approved — an untraceable claim is exactly what this guards against.
                return {
                    "status": "UNVERIFIED",
                    "claim_checked": claim,
                    "reason": (f"'{key}' is listed as approved but its source line could not "
                               f"be located in the brand documents. Treating as unverified."),
                    "approved_alternative": "Verify the claim against data/brand_docs/ or remove it.",
                }
            return {
                "status": "APPROVED",
                "claim_checked": claim,
                "reason": "Claim verified against brand source documents.",
                "verified_text": approved.text,
                "source_quote": approved.source_quote,
                "source_doc": approved.source_doc,
            }

    return {
        "status": "UNVERIFIED",
        "claim_checked": claim,
        "reason": "Claim not found in approved brand guidelines. Requires human review before use.",
        "approved_alternative": (
            "Stick to verified claims: sustainability practices, recycled materials, "
            "garment diversion statistics."
        ),
    }


def main() -> int:
    problems = audit_approved_claims()
    print(f"\nAuditing {len(APPROVED_CLAIMS)} approved claims against "
          f"{len(_load_docs())} brand documents\n")
    for key, claim in APPROVED_CLAIMS.items():
        bad = [p for p in problems if p["key"] == key]
        mark = "✗" if bad else "✓"
        print(f"  [{mark}] {key:<12} {claim.text[:58]}")
        print(f"       ↳ {claim.source_doc}: \"{claim.source_quote[:66]}\"")
        for p in bad:
            print(f"       !! {p['issue']}")
    if problems:
        print(f"\n✗ {len(problems)} approved claim(s) cannot be traced to a brand document.")
        print("  These will not be stamped APPROVED at runtime.\n")
        return 1
    print(f"\n✓ Every approved claim traces to a brand document.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
