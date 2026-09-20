# Brand Trust Agent

A verification pipeline for brand-safe campaign generation, instrumented with **Arize AX**.
Built for Verdant, a sustainable activewear brand, as a demonstration of claim
grounding, guardrails and per-step cost observability.

## Architecture

![Architecture Diagram](architecture.svg)

## What It Does

A **pipeline**, not an agent system. The control flow is fixed in `pipeline/run.py` — no model decides what happens next. The only branches are on data: did verification pass, did routing publish.

| Step | What it does | Span kind |
|------|--------------|-----------|
| **Retrieve** | ChromaDB over the brand docs. Plain function, no model call. | `RETRIEVER` |
| **Generate** | One `gpt-5.6-luna` call, structured output: copy **plus an explicit `claims[]` list**, each claim quoting its source. | `LLM` |
| **Verify** | Deterministic checks (free), then the `gpt-5.6-terra` judge only on survivors. | `GUARDRAIL` → `TOOL` + `LLM` |
| **Repair** | One rewrite against the contradicting source line. Hard cap of one attempt. | `LLM` |
| **Re-verify** | The repair gets checked too. Nothing is trusted for having been rewritten. | `GUARDRAIL` |
| **Route** | Publish with receipts, flag to human, or sample for audit. Plain function. | `CHAIN` |
| **Assets** | Veo video, only once routing publishes. | `TOOL` |
| **Audience** | Simulated reaction to what published, for the next cycle's brief. Series only. | `LLM` |

```
campaign-run                  CHAIN   (root)
├── retrieve                  RETRIEVER
├── generate                  LLM
├── verify                    GUARDRAIL
│   ├── deterministic-checks  TOOL
│   └── grounding-judge       LLM        (only if deterministic passed)
├── repair                    LLM        (only on failure)
├── re-verify                 GUARDRAIL  (only on failure)
├── route                     CHAIN
├── assets                    TOOL       (only when routing publishes)
└── audience-comments         LLM        (series only, on published cycles)
```

**The guardrail** (`evals/grounding_check.py`) compares the copy against *the chunks actually retrieved*, not the full corpus — so a retrieval failure is visible rather than silently papered over. It catches claims that quietly broaden a narrow source ("our Portugal factory is Fair Trade certified" → "our manufacturing is Fair Trade certified"), and the halt reason names the specific claim.

**The claim table checks itself.** Every entry in `brand_policy.py` carries the verbatim line from `data/brand_docs/` it rests on, verified at first use. An entry whose source cannot be located is not treated as approved, and `python3 -m brand_policy` exits non-zero. This exists because the table once carried three claims nothing supported — two about recycled packaging, which no brand document mentions at all, and "45,000 garments diverted annually" when the source says 45,312 *cumulative since 2021*. A claim stamped "(verified)" that no document supports is worse than an unchecked model, because the policy layer lends it a credential the evidence does not.

The retrieval-quality score is **reported, not gating**. It is built from retrieval signals and never sees the generated text, so a perfect retrieval plus a fabricating model scored high and passed.

**Routing and the audit sample.** A flagged campaign goes to a human and does not publish. A passing campaign publishes — *including when it is sampled*. Sampling audits what normally happens, so a sampled item takes the normal path and the review copy goes out in parallel. Holding 10% of approved content back would create a second behaviour and mean observing that instead.

**Cost.** Every step's token usage is measured and priced; `cost.step_tokens` / `cost.step_usd` sit on each span and the total on the root, so cost per campaign is visible broken down by step. A clean run is roughly 7,000 tokens and $0.0035; one that repairs is nearer 8,500 and $0.0042.

Going from top-4 to top-8 roughly doubled the retrieved source text (1,838 → 3,876 chars, about +510 tokens), which is carried by both the generate prompt and the judge prompt. That is about **+$0.0005 per judge call** and +$0.00005 per generate call — worst case around **+$0.0011 per campaign**. Observed end-to-end mean over three fixed briefs went $0.00255 → $0.00343, though that comparison is noisy: all three of the top-8 runs were caught by the deterministic layer, so the judge never ran.

Each step resolves to one of three states, and the last two are never collapsed:

- **known** — usage captured and priced
- **free** — genuinely no model call, e.g. a deterministic catch
- **unknown** — usage could not be captured

An unknown step withholds the run total rather than under-reporting it, and the end-to-end eval excludes unknown runs from its mean instead of averaging them in as zero. Long-context tier crossings are flagged, never silently priced at the cheaper rate.

## Evals

Four scripts, four different questions. They were one script; that was the problem.

```bash
python3 -m evals.eval_retrieval      # Did search find the right chunks?
python3 -m evals.eval_end_to_end     # Does the pipeline publish anything bad?
python3 -m evals.eval_judge          # Does the text judge agree with a fixed label?
python3 -m evals.eval_video_judge    # Does the video judge agree with a fixed label?
python3 -m evals.eval_video_judge --repeat 5   # …and does it say the same thing twice?
```

**Retrieval eval** — deterministic, no judge, costs a fraction of a cent. Checks whether the facts a brief needs are actually in the retrieved chunks, using the golden dataset's `ground_truth_answer_contains` as ground truth.

Four configurations, measured one variable at a time. Chunk size 600 throughout.

| config | mean recall | briefs incomplete | mean grounding | scored "high" | **"high" AND incomplete** |
|---|---|---|---|---|---|
| top-4, query template *(until 2026-09-20)* | 70.0% | 9 / 20 | — | 20 / 20 | **9** |
| **top-8, query template** *(current)* | **90.0%** | **3 / 20** | 0.760 | 15 / 20 | **2** |
| top-8, bare brief as query | 91.2% | 3 / 20 | 0.524 | 2 / 20 | **0** |
| top-8, LLM query expansion | 87.5% | 4 / 20 | 0.844 | 19 / 20 | **4** |

Raising `n_results` from 4 to 8 recovered 20 points of recall and cut incomplete briefs by two thirds. The other two rows are the interesting ones.

**The bare brief's zero is an artefact, not a fix.** It retrieves marginally better, but mean grounding collapses from 0.760 to 0.524 and only 2 of 20 briefs clear the "high" threshold at all. The same three briefs still miss facts — they now score 0.46 instead of 0.72. Nothing is *eligible* to be high-and-incomplete. The metric didn't improve; its denominator vanished.

**Query expansion is the clearest evidence the score cannot gate anything.** Only the phrasing of the query changed — same documents, same chunking, same index. Mean grounding rose **0.524 → 0.844** and "high" briefs went 2 → 19, while **recall fell** 91.2% → 87.5% and high-and-incomplete *doubled* to 4.

Expansion works by making the query look more like the documents — material names, section headings, product nouns — so cosine similarity rises because the *query* moved, not because the right passages were found. **The score went up 0.32 while retrieval got worse.** A number that can be inflated by rewording the question, with the corpus untouched, cannot be used to decide whether a campaign is safe to publish. It is reported, never gating; `VERIFY` reads the generated text against the retrieved chunks instead.

Expansion also has a shape to its failure: it helps product and material briefs and actively harms brand-voice ones. *"Brand voice compliance review"* expanded to product vocabulary and lost the tone-prohibition passages entirely — 0% recall at 0.88 grounding. Cost was $0.000136 per campaign, one cheap call, for negative recall.

**The finding survives every config.** In the shipping configuration two briefs still miss a required fact while scoring "high": id 6 at 0.72 (missing the 45,000 Take Back figure) and id 16 at 0.72 (missing the Watershed Jacket's `100% recycled nylon shell`, at **0% recall**). More retrieval made the gap smaller, never absent.

**End-to-end eval** — runs the real pipeline. The number that matters is **escapes**: campaigns routed to publish that still contain a claim the brief's row lists as prohibited. Its inputs move whenever the generator changes, so it is a snapshot, not a measurement.

**Judge eval** — frozen text, fixed hand-assigned labels, no pipeline at all. Sixteen cases in `data/judge_eval_set.csv` that never change, so a prompt or model change is measured against a fixed target.

> The labels in `data/judge_eval_set.csv` were hand-assigned against the brand documents and **need review by someone who owns the brand voice**. Read them before trusting the numbers.

Keeping these separate matters. When they were one script, the judge was scored against `expected_hallucination` — a column describing how risky a *query* looked when the dataset was written, not whether the *generated text* contained a false claim. That produced TPR 0.00 for a judge that was right on every disputed row.

## Series Mode

The app runs a campaign series — up to five cycles — the way an autonomous content calendar would.

Each cycle's brief is the original brief, plus the copy that was actually approved and published last cycle, plus what the audience said about it, plus every correction earned so far. Nothing else carries forward:

- **Brand facts are re-retrieved from source every cycle.** No chunk, fact or summary from a prior cycle is reused — `retrieve()` has nowhere to pass one.
- **Rejected drafts are never passed forward.** A cycle that fails the grounding check publishes nothing, so it contributes no copy and collects no comments.
- **No engagement metrics.** No CTR, no engagement rate, no view count, no "insight" telling the next cycle what to lean into.

### Audience comments are simulated

**They are generated by `gpt-5.6-luna`, not collected from anyone.** There is no real audience and no real engagement data anywhere in this project. Each published cycle gets one cheap model call that writes three comments responding to *that campaign's own published copy* — so a Take Back campaign draws comments about the Take Back Program.

The escalation is authored too. `_escalation(cycle)` instructs the model to keep early cycles accurate and to introduce plausible over-reaches later: rounding a qualified number up, assuming a certification the copy implied, generalising one certified factory to all of them. So the drift the series demonstrates is *scripted pressure*, not emergent behaviour. It is realistic in shape, not in provenance.

It runs only on cycles that publish — nobody comments on something they never saw — and is skipped on the final cycle, where the comments would feed nothing. Off by default (`simulate_audience=False`), so the evals never pay for it.

### The brief uses comments two ways

What the audience **responded to** decides what the next campaign is about; what they **believe that isn't true** decides what it must not claim. Enthusiasm is a signal about subject matter, never about facts.

### Corrections carry forward

Every flagged cycle records the claim, the source line it collided with, and the correction. The next cycle inherits all of them — with the source line, so the model knows *why*, not just *what*. The UI counts corrections per cycle and compares the first half of the series to the second: if they decline, the system is carrying its mistakes forward rather than relearning them.

**The grounding check runs every cycle, before any video is generated.** On a flag the copy is rewritten once against the source line, then re-checked once. Two attempts, hard cap. If the second check fails, the cycle publishes nothing and the reviewer sees both drafts, the flagged claim, the source line that contradicts it, and the reason.

## Video prompts

Two things the video model gets wrong unless you stop it, both handled in `build_video_prompt`:

**It renders concepts literally.** Asked for "recycled materials" it produced a shot of someone running past a recycling bin. So the campaign copy is never passed through — the theme is translated to physical, filmable content first (material quality → fibres catching light; circularity → the same garment worn, returned, worn again), and `find_banned_terms()` rejects ~35 abstract words with one retry and a known-clean fallback. `green` stays allowed: it is a colour in the brand palette, not an idea.

**It produces states, not shots.** "Someone running" has no beat. Every prompt is built as subject + the product clearly in frame + *the turn* + camera + light, with the turn mandatory — something released, revealed, or handed on inside the five seconds. `has_turn()` checks for it and retries once if it is missing.

That ban list is **video-only**. The copy prompt opens with "a sustainable activewear brand" and that is correct; the asymmetry is documented in both files so nobody reconciles it later.

### The video guardrail

The shot description is checked before a frame is generated, in the same two-layer shape as the copy guardrail. A refused shot **holds the whole campaign** — the copy does not ship on its own.

**Layer 1 — deterministic, no model call.** The same checks the copy goes through, run over the shot description: prohibited phrases, statistics outside the approved set, competitor names. One implementation, shared; only the wording of the failure differs, so a blocked shot says *"Video blocked: the shot description cites…"* rather than describing a halted campaign.

One exception applies to shots only: numbers attached to a measurement unit are ignored. `35mm`, `24fps`, `f/2.8`, `5 seconds` are cinematography, not brand claims, and flagging them blocked every video the system ever tried to make. A bare number in a shot — *"a wall of 400 reclaimed bottles"* — is still a claim and is still flagged.

**Layer 2 — the visual-claim judge**, on shots that pass layer 1. It asks a different question from the text judge: copy is judged on what it says, a shot on what a viewer would conclude from seeing it. A rooftop blanketed in solar panels asserts the factory runs on solar; the sources say 68% of factory *electricity*. `VIDEO_JUDGE_TEMPLATE` is version-controlled next to the text judge.

It runs on `gpt-5.6-terra` — **a different model from the `gpt-5.6-luna` generator**, so the checker does not inherit the writer's blind spots.

Both layers **fail closed**: a trip, or a judge that cannot run, blocks generation.

### The video judge is measured, and it is not stable

Five hand-labelled shots in `data/video_judge_eval_set.csv`. Current result: **2 of 5 agreement with the labels** — and repeated runs over the *same* shots produce *different verdicts*.

`--repeat 5` measured that directly, without reading labels: **3 of 5 cases fully stable, 92% mean consistency**, with two cases flipping between `supported` and `unsupported` across identical inputs. The judge cannot be pinned to temperature 0 — `gpt-5.6-terra` accepts only its default — so some variance is expected, but not this much.

Treat the video judge as an early signal, not a gate you would trust unattended. The deterministic layer beneath it is not affected: it is string matching and returns the same answer every time.

## Demo Scenarios

Use the sidebar toggle to switch between scenarios:

1. ✅ **Normal run** — retrieval, generation, verification, publish with receipts
2. ⚠️ **Weak retrieval** — one chunk instead of four, so claims may have no retrieved source to check against
3. 🔴 **Prompt injection** — an adversarial document tries to hijack the output; Verify reads the generated text, not the instruction

The old "trust gap" and "trust-aware" scenarios are gone. They existed so one agent could inherit another's confidence score; this is a pipeline and there is nothing to propagate between.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/rr0214/campaign-studio
cd campaign-studio
pip install -r requirements.txt
```

### 2. Configure credentials

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "..."        # For Veo video generation (optional)
ARIZE_API_KEY = "..."
ARIZE_SPACE_ID = "..."
```

You need:
- **OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Arize AX Space ID + API Key** — [app.arize.com](https://app.arize.com) → Settings → API Keys
- **Google API key** — [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) *(optional — video generation only)*

> The app runs fully without a Google API key. The Assets step still writes the shot description; only the video call is skipped. Retrieval, generation, the guardrail, repair, routing, tracing and cost accounting all work without it.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 4. Checks

```bash
python3 -m brand_policy          # every approved claim traces to a document (exits 1 if not)
python3 -m tests.test_pipeline   # pipeline behaviour, mocked — no network, no spend
python3 -m tests.test_regressions # the two review defects, so they stay fixed
```

The first two need no API key. `test_regressions` covers the substring-match bug that
approved "100% recycled materials", and the accounting bug that priced an unmeasured
call at $0.00.

## Project Structure

```
campaign-studio/
├── app.py                              # Streamlit UI. Tracing initialised here.
├── brand_policy.py                     # Approved/prohibited claims. Self-checking; run as a module.
├── pipeline/
│   ├── retrieval.py                    # RETRIEVE + query derivation (one copy of the template)
│   ├── steps.py                        # GENERATE, VERIFY, REPAIR, ROUTE, ASSETS, AUDIENCE
│   ├── run.py                          # Fixed control flow + tracing. No model picks the next step.
│   ├── pricing.py                      # Token accounting, USD, known/free/unknown states
│   └── usage_capture.py                # Captures usage + request text for calls we don't own
├── evals/
│   ├── grounding_check.py              # Deterministic checks + the judge. The guardrail itself.
│   ├── eval_retrieval.py               # Did search find the right chunks? Deterministic.
│   ├── eval_end_to_end.py              # Does the pipeline publish anything bad? Escapes.
│   ├── eval_judge.py                   # Does the text judge agree with fixed labels?
│   ├── eval_video_judge.py             # Same for the video judge, plus --repeat stability
│   └── run_evals.py                    # Deprecated stub pointing at the four above
├── tests/
│   ├── test_pipeline.py                # Mocked pipeline behaviour
│   └── test_regressions.py             # Defects found in review, kept fixed
├── instrumentation/
│   └── arize_setup.py                  # Arize OTel setup. Pure instrumentation.
├── data/
│   ├── golden_dataset.csv              # 20 briefs for the retrieval and end-to-end evals
│   ├── judge_eval_set.csv              # 16 frozen cases for the text judge eval
│   ├── video_judge_eval_set.csv        # 5 frozen shots for the video judge eval
│   ├── LABELING_GUIDE.md               # How to label them, with the boundary rulings
│   └── brand_docs/
│       ├── verdant_brand_guide.txt
│       ├── verdant_products.txt
│       ├── verdant_sustainability.txt
│       └── verdant_poisoned.txt        # Prompt injection fixture. Excluded from evidence.
├── architecture.svg
└── requirements.txt
```

---
Built by Rebecca Riggs · 2026
