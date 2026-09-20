# Labeling guide — `judge_eval_set.csv`

How to decide whether a frozen case is `supported` or `unsupported`. This file is
the appeal court for the fixture: when a label is disputed, the ruling gets
written down here so the next disputed case is decided the same way.

## What the fixture measures

**The judge, and only the judge.** Given this text and these sources, is the
verdict right?

It does *not* measure what the pipeline ships. The deterministic layer blocks
some text before the judge ever sees it, and repair may rewrite it afterwards.
Those are facts about the cascade. Labeling a case by what survives to publication
mixes two subjects and makes the number mean nothing — which is exactly how the
first version of this eval reported TPR 0.00 for a judge that was right on every
disputed row.

## The standing rules

1. **Verifiable factual claims only.** Numbers, percentages, certifications,
   partnerships, materials, factory locations, programme results, dates. Marketing
   voice, tone and aspiration are not checkable and must never be flagged.
2. **Sources are the retrieved chunks, not the whole corpus.** A claim that is
   true elsewhere in the brand documents but absent from the chunks shown to the
   judge is `unsupported`. Retrieval failure is supposed to be visible.
3. **Broadening is unsupported.** If the source states something narrowly and the
   copy states it wider, the claim has changed.
4. **Reworded prohibitions count.** "Net zero", "climate positive", "we've
   balanced our footprint" all restate carbon neutrality.
5. **When genuinely uncertain, label `unsupported`.** Missing a false claim costs
   more than flagging a true one.

## Boundary rulings

Cases where the call was not obvious. Each one binds future cases of the same shape.

### Ruling 1 — partial-basis claims are broadening (case 4)

> Source: *"Renewable energy (solar): 68% of factory electricity as of Dec 2025"*
> Copy: *"Our Portugal factory … runs on 68% solar"*
> **Ruling: `unsupported`.**

Electricity is not everything a factory runs on, so "runs on 68% solar" claims
more than the source establishes.

*Generalises to:* when a source qualifies a figure by what it measures — by
weight, by electricity, by revenue — dropping that basis broadens the claim, even
though the number is unchanged.

### Ruling 2 — label the judge's subject, not the pipeline's behaviour (case 13)

> Source: *"We are NOT yet carbon neutral… Any claim of carbon neutrality in
> marketing materials is prohibited until audit is complete."*
> Copy: *"We are not yet carbon neutral, and we won't say otherwise until our
> audit is complete."*
> **Ruling: `supported`.**

The sentence is accurate and traces directly to the source. It was originally
labelled `unsupported` because the deterministic layer blocks the phrase "carbon
neutral" outright, so this text never ships — but that is a fact about the
cascade, not about the judge.

*Generalises to:* never label a case by what the deterministic layer or the
repair step would do to it. Label only whether the verdict on the text is correct.
The `deterministic_catch` column records which cases layer 1 short-circuits, and
`--judge-reachable-only` filters to the ones the judge actually sees in production.

## Adding a case

Keep sources realistic — paste real lines from `data/brand_docs/`. State the
label and one sentence of reasoning in `rationale`. If the call required a
judgement, add a ruling above rather than leaving the reasoning in the CSV.

**Do not relabel a case because the judge disagreed with it.** That fits the
target to the model and destroys the measurement. Change a label only when the
reasoning is wrong on its own terms, and record the ruling here.

## Ruling 3 — a certification shown without scope reads as brand-wide,
## and is supported only if a brand-wide certification exists

Video case 2 (an unnamed seal filling the frame) was first labelled unsupported,
reasoning from Fair Trade USA (Portugal only) and B Corp (pending). That was
wrong on its own terms: OEKO-TEX Standard 100 covers all textiles, so an unnamed
seal on a garment tag does have brand-wide support. Relabelled supported.

*Generalises to:* check every certification the documents grant before ruling a
depicted seal unsupported. Scope-limited certifications do not make an unnamed
seal false if any brand-wide certification exists.

## Ruling 4 — depicted totality is a claim; depicted behaviour is not

Video case 3 ("each one lifted into a single bin marked for reuse, nothing left
behind") was first labelled supported. Wrong: the sources state 8% of returns are
landfilled, so the shot contradicts them outright. Relabelled unsupported.

Video case 5 (a worn jacket handed from one runner to another) stays supported.
The judge flagged it, reasoning a viewer would conclude Verdant "operates or
documents a direct garment-repair and peer-to-peer reuse handoff program". That
is the judge reading a human gesture as an institutional claim, and it is a
known weakness of VIDEO_JUDGE_TEMPLATE — kept in the set deliberately.

*Generalises to:* a shot asserting completeness ("every", "nothing left behind")
is checkable against the documents. A shot depicting a person doing something is
not an assertion that a programme exists.

## Provenance

The initial 16 cases and their labels were drafted by Claude against the brand
documents on 2026-09-19, then reviewed. Rulings 1 and 2 were issued by the
repository owner. Labels still warrant review by someone who owns the brand voice.
